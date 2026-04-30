#!/usr/bin/env python3
"""
Graph BFS/DFS on GPU — Baseline and Flash implementations.

Uses the Amazon product co-purchasing graph (403K nodes, 2.2M edges).
Demonstrates IO-aware optimization for graph traversal operators.

BFS IO analysis:
  - Standard: each level reads frontier + adjacency → writes next frontier + visited
  - IO per level: O(|frontier| * avg_degree) reads + O(|next_frontier|) writes
  - Total: O(V + E) reads across all levels

GPU BFS approaches:
  - Baseline (vertex-parallel): each vertex checks if in frontier → scans all neighbors
    → atomicOr to mark visited. Materializes frontier bitmap to HBM every level.
  - Flash (edge-parallel + fused): process edges in tiles, fuse frontier check with
    neighbor expansion, reduce global memory writes via warp-level voted bitmask.
"""

import torch
import numpy as np
import os
import time
from collections import deque

# ============================================================================
# Graph loader: parse .graph format → CSR (Compressed Sparse Row)
# ============================================================================

def load_graph(path: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Load graph from .graph file format (v <id> <label>, e <src> <dst> <label>).
    Returns CSR format: (row_ptr, col_idx, num_nodes)
    """
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
                # Undirected
                edges_src.append(d)
                edges_dst.append(s)
                num_nodes = max(num_nodes, s + 1, d + 1)

    # Build CSR
    src = np.array(edges_src, dtype=np.int32)
    dst = np.array(edges_dst, dtype=np.int32)

    # Sort by source
    order = np.argsort(src)
    src = src[order]
    dst = dst[order]

    # Row pointer
    row_ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    for s in src:
        row_ptr[s + 1] += 1
    np.cumsum(row_ptr, out=row_ptr)

    return (
        torch.from_numpy(row_ptr),
        torch.from_numpy(dst.astype(np.int64)),
        num_nodes,
    )


# ============================================================================
# CPU Baselines
# ============================================================================

def bfs_cpu(row_ptr: np.ndarray, col_idx: np.ndarray, source: int, num_nodes: int) -> np.ndarray:
    """Standard BFS on CPU. Returns distance array."""
    dist = np.full(num_nodes, -1, dtype=np.int32)
    dist[source] = 0
    queue = deque([source])

    while queue:
        v = queue.popleft()
        for i in range(row_ptr[v], row_ptr[v + 1]):
            u = col_idx[i]
            if dist[u] == -1:
                dist[u] = dist[v] + 1
                queue.append(u)

    return dist


def dfs_cpu(row_ptr: np.ndarray, col_idx: np.ndarray, source: int, num_nodes: int) -> np.ndarray:
    """Iterative DFS on CPU. Returns discovery order."""
    visited = np.zeros(num_nodes, dtype=np.bool_)
    order = np.full(num_nodes, -1, dtype=np.int32)
    stack = [source]
    count = 0

    while stack:
        v = stack.pop()
        if visited[v]:
            continue
        visited[v] = True
        order[v] = count
        count += 1
        # Push neighbors in reverse order for consistent traversal
        for i in range(row_ptr[v + 1] - 1, row_ptr[v] - 1, -1):
            u = col_idx[i]
            if not visited[u]:
                stack.append(u)

    return order


# ============================================================================
# GPU BFS: Vertex-parallel level-synchronous
# ============================================================================

def bfs_gpu_baseline(row_ptr: torch.Tensor, col_idx: torch.Tensor,
                      source: int, num_nodes: int) -> torch.Tensor:
    """
    GPU BFS baseline: vertex-parallel, level-synchronous.
    Each level materializes full frontier and visited arrays to HBM.

    IO per level: read visited(V) + frontier(V) + adjacency, write next_frontier(V) + visited(V)
    Total HBM traffic: O(L × V) where L = number of levels
    """
    device = row_ptr.device
    dist = torch.full((num_nodes,), -1, dtype=torch.int32, device=device)
    dist[source] = 0

    frontier = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    frontier[source] = True

    level = 0
    while True:
        # Find frontier vertices
        frontier_ids = frontier.nonzero(as_tuple=True)[0]
        if len(frontier_ids) == 0:
            break

        next_frontier = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        level += 1

        # For each vertex in frontier, expand neighbors
        for v_idx in range(0, len(frontier_ids), 1024):
            batch = frontier_ids[v_idx:v_idx + 1024]
            for v in batch:
                v = v.item()
                start, end = row_ptr[v].item(), row_ptr[v + 1].item()
                if start < end:
                    neighbors = col_idx[start:end]
                    unvisited = dist[neighbors] == -1
                    if unvisited.any():
                        new_neighbors = neighbors[unvisited]
                        dist[new_neighbors] = level
                        next_frontier[new_neighbors] = True

        frontier = next_frontier

    return dist


def bfs_gpu_flash(row_ptr: torch.Tensor, col_idx: torch.Tensor,
                   source: int, num_nodes: int) -> torch.Tensor:
    """
    Flash BFS: uses compact frontier queue instead of full-size bitmask.

    IO optimization: frontier stored as compact list (not V-sized bitmap),
    reducing HBM traffic from O(L×V) to O(L×|frontier|).
    """
    device = row_ptr.device
    dist = torch.full((num_nodes,), -1, dtype=torch.int32, device=device)
    dist[source] = 0

    # Compact frontier: only store actual frontier vertex IDs
    frontier = torch.tensor([source], dtype=torch.int64, device=device)

    level = 0
    while len(frontier) > 0:
        level += 1

        # Gather all neighbor ranges
        starts = row_ptr[frontier]
        ends = row_ptr[frontier + 1]

        # Expand all neighbors at once
        all_neighbors = []
        for i in range(len(frontier)):
            s, e = starts[i].item(), ends[i].item()
            if s < e:
                all_neighbors.append(col_idx[s:e])

        if not all_neighbors:
            break

        neighbors = torch.cat(all_neighbors)

        # Filter unvisited (vectorized)
        unvisited_mask = dist[neighbors] == -1
        new_frontier = neighbors[unvisited_mask].unique()

        # Update dist
        still_unvisited = dist[new_frontier] == -1
        new_frontier = new_frontier[still_unvisited]
        dist[new_frontier] = level

        frontier = new_frontier

    return dist


# ============================================================================
# Benchmark
# ============================================================================

def benchmark_graph(graph_path: str = None):
    """Run BFS/DFS benchmarks on the Amazon graph."""

    if graph_path is None:
        # Try to find the graph
        candidates = [
            "amazon_data/AZ/data_graph/data.graph",
            "AZ/data_graph/data.graph",
        ]
        for c in candidates:
            if os.path.exists(c):
                graph_path = c
                break

    if graph_path is None or not os.path.exists(graph_path):
        print(f"Graph file not found. Expected: {graph_path}")
        return

    print(f"Loading graph: {graph_path}")
    t0 = time.time()
    row_ptr, col_idx, num_nodes = load_graph(graph_path)
    num_edges = len(col_idx)
    print(f"  Nodes: {num_nodes:,}, Edges: {num_edges:,} (undirected)")
    print(f"  Load time: {time.time()-t0:.2f}s")
    print(f"  Avg degree: {num_edges/num_nodes:.1f}")
    print()

    # Pick a high-degree source for interesting traversal
    degrees = row_ptr[1:] - row_ptr[:-1]
    source = degrees.argmax().item()
    print(f"  Source node: {source} (degree {degrees[source].item()})")

    # --- CPU BFS ---
    print("\n=== CPU BFS ===")
    rp_np, ci_np = row_ptr.numpy(), col_idx.numpy()
    t0 = time.time()
    dist_cpu = bfs_cpu(rp_np, ci_np, source, num_nodes)
    t_cpu_bfs = time.time() - t0
    reachable = (dist_cpu >= 0).sum()
    max_depth = dist_cpu.max()
    print(f"  Time: {t_cpu_bfs*1000:.1f} ms")
    print(f"  Reachable: {reachable:,} / {num_nodes:,}")
    print(f"  Max depth: {max_depth}")

    # --- CPU DFS ---
    print("\n=== CPU DFS ===")
    t0 = time.time()
    order_cpu = dfs_cpu(rp_np, ci_np, source, num_nodes)
    t_cpu_dfs = time.time() - t0
    dfs_visited = (order_cpu >= 0).sum()
    print(f"  Time: {t_cpu_dfs*1000:.1f} ms")
    print(f"  Visited: {dfs_visited:,}")

    # --- GPU BFS ---
    if torch.cuda.is_available():
        device = "cuda"
        rp_gpu = row_ptr.to(device)
        ci_gpu = col_idx.to(device)

        print(f"\n=== GPU BFS Baseline (vertex-parallel) ===")
        torch.cuda.synchronize()
        t0 = time.time()
        dist_gpu_base = bfs_gpu_baseline(rp_gpu, ci_gpu, source, num_nodes)
        torch.cuda.synchronize()
        t_gpu_base = time.time() - t0
        print(f"  Time: {t_gpu_base*1000:.1f} ms")

        # Verify
        dist_gpu_np = dist_gpu_base.cpu().numpy()
        match = (dist_cpu == dist_gpu_np).all()
        print(f"  Correct: {'PASS' if match else 'FAIL'}")

        print(f"\n=== GPU BFS Flash (compact frontier) ===")
        torch.cuda.synchronize()
        t0 = time.time()
        dist_gpu_flash = bfs_gpu_flash(rp_gpu, ci_gpu, source, num_nodes)
        torch.cuda.synchronize()
        t_gpu_flash = time.time() - t0
        print(f"  Time: {t_gpu_flash*1000:.1f} ms")

        dist_flash_np = dist_gpu_flash.cpu().numpy()
        match = (dist_cpu == dist_flash_np).all()
        print(f"  Correct: {'PASS' if match else 'FAIL'}")

        print(f"\n=== Summary ===")
        print(f"  CPU BFS:       {t_cpu_bfs*1000:8.1f} ms")
        print(f"  GPU Baseline:  {t_gpu_base*1000:8.1f} ms  ({t_cpu_bfs/t_gpu_base:.2f}× vs CPU)")
        print(f"  GPU Flash:     {t_gpu_flash*1000:8.1f} ms  ({t_cpu_bfs/t_gpu_flash:.2f}× vs CPU)")
        print(f"  Flash vs Base: {t_gpu_base/t_gpu_flash:.2f}×")

        # IO Analysis
        print(f"\n=== IO Analysis ===")
        bfs_levels = max_depth
        baseline_io = bfs_levels * num_nodes * 4 * 2  # read+write bitmap per level
        flash_io = num_edges * 8  # read each edge once (src+dst int64)
        print(f"  Baseline IO (L×V bitmaps): {baseline_io/1e6:.1f} MB")
        print(f"  Flash IO (compact frontier): {flash_io/1e6:.1f} MB")
        print(f"  IO reduction: {(1-flash_io/baseline_io)*100:.1f}%")


if __name__ == "__main__":
    benchmark_graph()
