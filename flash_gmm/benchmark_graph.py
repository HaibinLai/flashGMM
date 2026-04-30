"""
Benchmark: C++/CUDA BFS on Amazon graph.

Compiles graph_bfs.cu via JIT, runs CPU BFS + GPU baseline + GPU flash,
compares correctness and speed.
"""
import torch
import time
import os
import sys
import numpy as np

# ============================================================================
# JIT compile the CUDA extension
# ============================================================================
from torch.utils.cpp_extension import load

csrc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_C = load(
    name="graph_bfs_native",
    sources=[os.path.join(csrc_dir, "graph_bfs.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)


# ============================================================================
# Graph loader (same as graph_bfs_dfs.py)
# ============================================================================
def load_graph(path: str):
    edges_src, edges_dst = [], []
    num_nodes = 0
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                num_nodes = max(num_nodes, int(parts[1]) + 1)
            elif parts[0] == 'e':
                s, d = int(parts[1]), int(parts[2])
                edges_src.append(s)
                edges_dst.append(d)
                edges_src.append(d)
                edges_dst.append(s)
                num_nodes = max(num_nodes, s + 1, d + 1)

    src = np.array(edges_src, dtype=np.int32)
    dst = np.array(edges_dst, dtype=np.int32)
    order = np.argsort(src)
    src, dst = src[order], dst[order]

    row_ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    for s in src:
        row_ptr[s + 1] += 1
    np.cumsum(row_ptr, out=row_ptr)

    return torch.from_numpy(row_ptr), torch.from_numpy(dst.astype(np.int64)), num_nodes


def bench_fn(fn, n_warmup=3, n_iter=10):
    """Benchmark a function (CPU or GPU)."""
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_iter):
        result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.time() - t0) / n_iter
    return elapsed, result


def main():
    graph_path = None
    for c in ["amazon_data/AZ/data_graph/data.graph", "AZ/data_graph/data.graph"]:
        if os.path.exists(c):
            graph_path = c
            break

    if not graph_path:
        print("Graph not found. Run from flash_gpu/ directory with amazon_data/ present.")
        return

    print(f"Loading: {graph_path}")
    row_ptr, col_idx, num_nodes = load_graph(graph_path)
    num_edges = len(col_idx)
    print(f"  V={num_nodes:,}  E={num_edges:,}  avg_deg={num_edges/num_nodes:.1f}")

    # Pick high-degree source
    degrees = row_ptr[1:] - row_ptr[:-1]
    source = degrees.argmax().item()
    print(f"  Source: {source} (degree {degrees[source].item()})")
    print()

    # ---- CPU BFS (C++) ----
    print("=== C++ CPU BFS ===")
    t_cpu, dist_cpu = bench_fn(lambda: _C.bfs_cpu(row_ptr, col_idx, source, num_nodes))
    reachable = (dist_cpu >= 0).sum().item()
    max_depth = dist_cpu.max().item()
    print(f"  Time: {t_cpu*1000:.2f} ms")
    print(f"  Reachable: {reachable:,}  Max depth: {max_depth}")

    # ---- CPU DFS (C++) ----
    print("\n=== C++ CPU DFS ===")
    t_dfs, order_cpu = bench_fn(lambda: _C.dfs_cpu(row_ptr, col_idx, source, num_nodes))
    dfs_visited = (order_cpu >= 0).sum().item()
    print(f"  Time: {t_dfs*1000:.2f} ms")
    print(f"  Visited: {dfs_visited:,}")

    if not torch.cuda.is_available():
        print("\nNo GPU available.")
        return

    # Move to GPU
    rp_gpu = row_ptr.cuda()
    ci_gpu = col_idx.cuda()

    # ---- GPU BFS Baseline ----
    print(f"\n=== CUDA GPU BFS Baseline (vertex-parallel, full scan) ===")
    t_base, dist_base = bench_fn(
        lambda: _C.bfs_gpu_baseline(rp_gpu, ci_gpu, source, num_nodes),
        n_warmup=2, n_iter=5)
    match_base = (dist_cpu == dist_base.cpu()).all().item()
    print(f"  Time: {t_base*1000:.2f} ms")
    print(f"  Correct: {'PASS' if match_base else 'FAIL'}")

    # ---- GPU BFS Flash ----
    print(f"\n=== CUDA GPU BFS Flash (compact frontier, edge-parallel) ===")
    t_flash, dist_flash = bench_fn(
        lambda: _C.bfs_gpu_flash(rp_gpu, ci_gpu, source, num_nodes),
        n_warmup=2, n_iter=5)
    match_flash = (dist_cpu == dist_flash.cpu()).all().item()
    print(f"  Time: {t_flash*1000:.2f} ms")
    print(f"  Correct: {'PASS' if match_flash else 'FAIL'}")

    # ---- Summary ----
    print(f"\n{'='*50}")
    print(f"  Graph: {num_nodes:,} nodes, {num_edges:,} edges")
    print(f"  BFS depth: {max_depth} levels")
    print(f"{'='*50}")
    print(f"  C++ CPU BFS:     {t_cpu*1000:8.2f} ms")
    print(f"  C++ CPU DFS:     {t_dfs*1000:8.2f} ms")
    print(f"  CUDA Baseline:   {t_base*1000:8.2f} ms  ({t_cpu/t_base:.2f}× vs CPU)")
    print(f"  CUDA Flash:      {t_flash*1000:8.2f} ms  ({t_cpu/t_flash:.2f}× vs CPU)")
    print(f"  Flash vs Base:   {t_base/t_flash:.2f}×")
    print(f"{'='*50}")

    # IO analysis
    baseline_io = max_depth * num_nodes * 4 * 3  # frontier+visited+next per level
    flash_io = reachable * degrees[:num_nodes].float().mean().item() * 8  # frontier×deg×8B
    print(f"  Baseline IO (L×V scans):     {baseline_io/1e6:.1f} MB")
    print(f"  Flash IO (compact frontier):  {flash_io/1e6:.1f} MB")
    print(f"  IO reduction: {(1-flash_io/baseline_io)*100:.1f}%")


if __name__ == "__main__":
    main()
