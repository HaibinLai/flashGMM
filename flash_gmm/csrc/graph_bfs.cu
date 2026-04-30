/*
 * Graph BFS/DFS: C++ CPU + CUDA GPU implementations
 *
 * CPU:
 *   - bfs_cpu: standard BFS with queue (single-thread)
 *   - dfs_cpu: iterative DFS with stack
 *
 * GPU:
 *   - bfs_gpu_baseline: vertex-parallel, level-synchronous
 *     Each level: scan ALL vertices, check frontier bitmap → expand
 *     IO: O(L × V) per-level full bitmap scan
 *
 *   - bfs_gpu_flash: edge-parallel with compact frontier
 *     Each level: only process frontier vertices → gather neighbors
 *     IO: O(Σ|frontier_i| × avg_degree) — proportional to actual work
 *
 * Input: CSR format (row_ptr, col_idx)
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include <stack>

// ============================================================================
// CPU BFS: standard queue-based
// ============================================================================
torch::Tensor bfs_cpu(
    torch::Tensor row_ptr,   // (V+1,) int64
    torch::Tensor col_idx,   // (E,) int64
    int64_t source,
    int64_t num_nodes
) {
    auto dist = torch::full({num_nodes}, -1, torch::kInt32);
    auto dist_ptr = dist.data_ptr<int32_t>();
    auto rp = row_ptr.data_ptr<int64_t>();
    auto ci = col_idx.data_ptr<int64_t>();

    dist_ptr[source] = 0;
    std::queue<int64_t> q;
    q.push(source);

    while (!q.empty()) {
        int64_t v = q.front(); q.pop();
        int32_t d = dist_ptr[v];
        for (int64_t i = rp[v]; i < rp[v + 1]; i++) {
            int64_t u = ci[i];
            if (dist_ptr[u] == -1) {
                dist_ptr[u] = d + 1;
                q.push(u);
            }
        }
    }
    return dist;
}

// ============================================================================
// CPU DFS: iterative stack-based
// ============================================================================
torch::Tensor dfs_cpu(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    int64_t source,
    int64_t num_nodes
) {
    auto order = torch::full({num_nodes}, -1, torch::kInt32);
    auto order_ptr = order.data_ptr<int32_t>();
    auto rp = row_ptr.data_ptr<int64_t>();
    auto ci = col_idx.data_ptr<int64_t>();

    std::vector<bool> visited(num_nodes, false);
    std::stack<int64_t> stk;
    stk.push(source);
    int32_t count = 0;

    while (!stk.empty()) {
        int64_t v = stk.top(); stk.pop();
        if (visited[v]) continue;
        visited[v] = true;
        order_ptr[v] = count++;
        for (int64_t i = rp[v + 1] - 1; i >= rp[v]; i--) {
            int64_t u = ci[i];
            if (!visited[u]) {
                stk.push(u);
            }
        }
    }
    return order;
}

// ============================================================================
// CUDA Kernel: BFS Baseline — vertex-parallel, full scan per level
//
// Each thread checks one vertex:
//   if frontier[v]: for each neighbor u: if !visited[u]: mark next_frontier[u]
//
// IO per level: read frontier(V) + visited(V) + adjacency, write next_frontier(V)
// Total: O(L × V) — wasteful when frontier is small
// ============================================================================
__global__ void bfs_baseline_kernel(
    const int64_t* __restrict__ row_ptr,
    const int64_t* __restrict__ col_idx,
    const int32_t* __restrict__ frontier,   // 1 if in frontier, 0 otherwise (V,)
    int32_t* __restrict__ next_frontier,    // output (V,)
    int32_t* __restrict__ dist,             // (V,)
    int32_t level,
    int64_t num_nodes
) {
    int64_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;

    if (frontier[v]) {
        for (int64_t i = row_ptr[v]; i < row_ptr[v + 1]; i++) {
            int64_t u = col_idx[i];
            if (dist[u] == -1) {
                dist[u] = level;
                next_frontier[u] = 1;
            }
        }
    }
}

torch::Tensor bfs_gpu_baseline(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    int64_t source,
    int64_t num_nodes
) {
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(row_ptr.device());
    auto dist = torch::full({num_nodes}, -1, opts_i32);
    auto frontier = torch::zeros({num_nodes}, opts_i32);
    auto next_frontier = torch::zeros({num_nodes}, opts_i32);

    dist.index_put_({source}, 0);
    frontier.index_put_({source}, 1);

    int threads = 256;
    int blocks = (num_nodes + threads - 1) / threads;
    int32_t level = 0;

    while (true) {
        level++;
        next_frontier.zero_();

        bfs_baseline_kernel<<<blocks, threads>>>(
            row_ptr.data_ptr<int64_t>(),
            col_idx.data_ptr<int64_t>(),
            frontier.data_ptr<int32_t>(),
            next_frontier.data_ptr<int32_t>(),
            dist.data_ptr<int32_t>(),
            level, num_nodes);

        cudaDeviceSynchronize();

        // Check if next_frontier is empty
        if (next_frontier.sum().item<int64_t>() == 0) break;

        // Swap
        std::swap(frontier, next_frontier);
    }
    return dist;
}

// ============================================================================
// CUDA Kernel: BFS Flash — frontier-driven, edge-parallel
//
// Only process vertices in the compact frontier list.
// Each thread handles one frontier vertex → expands its neighbors.
// Uses atomicCAS on dist[] to avoid race conditions.
//
// IO per level: read frontier_list(|F|) + adjacency(|F|×deg) + dist(neighbors)
//               write dist(new) + next_count
// Total: O(Σ|F_i| × avg_deg) — proportional to actual edges traversed
// ============================================================================
__global__ void bfs_flash_kernel(
    const int64_t* __restrict__ row_ptr,
    const int64_t* __restrict__ col_idx,
    const int64_t* __restrict__ frontier_list,  // compact list of frontier vertex IDs
    int64_t frontier_size,
    int32_t* __restrict__ dist,
    int64_t* __restrict__ next_list,            // output: new frontier vertices
    int32_t* __restrict__ next_count,           // atomic counter
    int32_t level,
    int64_t num_nodes
) {
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int64_t v = frontier_list[tid];
    for (int64_t i = row_ptr[v]; i < row_ptr[v + 1]; i++) {
        int64_t u = col_idx[i];
        // atomicCAS: only first thread to reach u marks it
        int32_t old = atomicCAS(&dist[u], -1, level);
        if (old == -1) {
            // Successfully claimed u — add to next frontier
            int32_t pos = atomicAdd(next_count, 1);
            if (pos < num_nodes) {
                next_list[pos] = u;
            }
        }
    }
}

torch::Tensor bfs_gpu_flash(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    int64_t source,
    int64_t num_nodes
) {
    auto device = row_ptr.device();
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);

    auto dist = torch::full({num_nodes}, -1, opts_i32);
    dist.index_put_({source}, 0);

    // Compact frontier list (not V-sized bitmap!)
    auto frontier_list = torch::tensor({source}, opts_i64);
    auto next_list = torch::empty({num_nodes}, opts_i64);
    auto next_count = torch::zeros({1}, opts_i32.dtype(torch::kInt32));

    int threads = 256;
    int32_t level = 0;

    while (frontier_list.size(0) > 0) {
        level++;
        int64_t fsize = frontier_list.size(0);
        int blocks = (fsize + threads - 1) / threads;

        next_count.zero_();

        bfs_flash_kernel<<<blocks, threads>>>(
            row_ptr.data_ptr<int64_t>(),
            col_idx.data_ptr<int64_t>(),
            frontier_list.data_ptr<int64_t>(),
            fsize,
            dist.data_ptr<int32_t>(),
            next_list.data_ptr<int64_t>(),
            next_count.data_ptr<int32_t>(),
            level, num_nodes);

        cudaDeviceSynchronize();

        int32_t ncount = next_count.item<int32_t>();
        if (ncount == 0) break;

        frontier_list = next_list.slice(0, 0, ncount).clone();
    }
    return dist;
}

// ============================================================================
// Pybind11
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Graph BFS/DFS: C++ CPU + CUDA GPU";
    m.def("bfs_cpu", &bfs_cpu, "CPU BFS (queue-based)");
    m.def("dfs_cpu", &dfs_cpu, "CPU DFS (stack-based)");
    m.def("bfs_gpu_baseline", &bfs_gpu_baseline,
          "GPU BFS baseline: vertex-parallel, full scan per level. IO: O(L×V)");
    m.def("bfs_gpu_flash", &bfs_gpu_flash,
          "GPU BFS flash: frontier-driven, edge-parallel. IO: O(Σ|F|×deg)");
}
