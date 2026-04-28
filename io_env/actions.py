"""
Design Action Space for IO-aware operator optimization.

Each action transforms a ComputationGraph to reduce IO,
with clear symbolic semantics the agent can reason about.
"""

from __future__ import annotations
from .dsl import ComputationGraph, TensorSpec, Op, FusedOp, TilingSpec, OnlineStateSpec
import copy


# Registry of known online algorithms
ONLINE_ALGORITHMS = {
    "online_logsumexp": {
        "description": "Online log-sum-exp: maintain (running_max, running_sum_exp) per point",
        "state_vars": {"running_max": "scalar", "running_sum_exp": "scalar"},
        "input_reduce_dim": "K",     # reduces over K
        "output_shape_rule": lambda shape: tuple(s for s in shape if s != "K"),
        "applicable_computes": ["log_likelihood+logsumexp", "logsumexp"],
    },
    "online_argmin": {
        "description": "Online argmin: maintain (running_min, running_idx) per point",
        "state_vars": {"running_min": "scalar", "running_idx": "scalar"},
        "input_reduce_dim": "K",
        "output_shape_rule": lambda shape: tuple(s for s in shape if s != "K"),
        "applicable_computes": ["distance+argmin", "argmin"],
    },
    "online_softmax": {
        "description": "Online softmax: maintain (running_max, running_sum_exp, running_output) per point",
        "state_vars": {"running_max": "scalar", "running_sum_exp": "scalar", "running_o": "vector_d"},
        "input_reduce_dim": "K",
        "output_shape_rule": lambda shape: tuple(s for s in shape if s != "K"),
        "applicable_computes": ["attention_score+softmax", "softmax"],
    },
    "online_welford": {
        "description": "Online Welford: maintain (count, mean, M2) for numerically stable variance",
        "state_vars": {"count": "scalar", "mean": "scalar", "M2": "scalar"},
        "input_reduce_dim": "N",
        "output_shape_rule": lambda shape: ("K", "d") if len(shape) > 1 else ("K",),
        "applicable_computes": ["variance", "batchnorm_stats"],
    },
}


class DesignActions:
    """
    Actions an agent can take to transform a ComputationGraph.
    Each action returns a new graph (original is not modified).
    """

    @staticmethod
    def fuse_ops(graph: ComputationGraph, op1_name: str, op2_name: str,
                 fused_name: str | None = None) -> ComputationGraph:
        """
        Fuse two sequential operations, eliminating the intermediate tensor.

        The intermediate (output of op1, input of op2) no longer materializes to HBM.
        Combined FLOPs = op1.flops + op2.flops.
        IO savings = 2 × size(intermediate) (write + read eliminated).
        """
        new_graph = graph.clone()
        ops = new_graph.operations

        idx1 = next(i for i, op in enumerate(ops) if op.name == op1_name)
        idx2 = next(i for i, op in enumerate(ops) if op.name == op2_name)
        op1, op2 = ops[idx1], ops[idx2]

        # The intermediate is op1's output
        intermediate_name = op1.output_name

        # Combined reads = op1.reads + (op2.reads - intermediate)
        combined_reads = list(op1.reads) + [r for r in op2.reads if r != intermediate_name]

        # Combined FLOPs
        combined_flops_fn = lambda p, f1=op1.flops_fn, f2=op2.flops_fn: f1(p) + f2(p)

        name = fused_name or f"fused_{op1_name}_{op2_name}"
        fused = FusedOp(
            name=name,
            reads=combined_reads,
            computes=f"{op1.computes}+{op2.computes}",
            flops_fn=combined_flops_fn,
            output=op2.output,
            output_name=op2.output_name,
            notes=f"Fused {op1_name} + {op2_name}, eliminated '{intermediate_name}' materialization",
        )

        # Replace the two ops with the fused op
        new_ops = [op for i, op in enumerate(ops) if i not in (idx1, idx2)]
        new_ops.insert(min(idx1, idx2), fused)
        new_graph.operations = new_ops

        # Rewrite any remaining references to the eliminated intermediate.
        # Other ops that read the intermediate now read the fused op's inputs directly.
        # Since the fused op absorbs the intermediate's computation, downstream ops
        # that also read the intermediate should instead read the fused op's *output*
        # (which has the same name as op2's output). If they truly needed the
        # intermediate's value, that's a semantic issue; here we simply remove
        # the dangling reference since the intermediate is no longer produced.
        for op in new_graph.operations:
            if op is not fused and intermediate_name in op.reads:
                op.reads = [r for r in op.reads if r != intermediate_name]

        return new_graph

    @staticmethod
    def apply_online_algorithm(graph: ComputationGraph, target_op: str,
                                algorithm: str,
                                tiling: TilingSpec | None = None) -> ComputationGraph:
        """
        Replace a reduce operation with an online/streaming algorithm.

        Instead of materializing the full intermediate matrix, maintain
        a small online state (registers) and process data in tiles.

        Output shape shrinks from (N, K) to (N,) or similar.
        """
        if algorithm not in ONLINE_ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(ONLINE_ALGORITHMS.keys())}")

        algo_spec = ONLINE_ALGORITHMS[algorithm]
        new_graph = graph.clone()
        ops = new_graph.operations

        idx = next(i for i, op in enumerate(ops) if op.name == target_op)
        old_op = ops[idx]

        online_state = OnlineStateSpec(
            variables=algo_spec["state_vars"],
            algorithm=algorithm,
            output_shape=algo_spec["output_shape_rule"](old_op.output.shape),
        )

        # The output shape is reduced (e.g., (N, K) -> (N,))
        new_output = TensorSpec(
            shape=online_state.output_shape,
            dtype=old_op.output.dtype,
            storage="HBM",  # final result still goes to HBM, but it's O(N) not O(NK)
        )

        new_op = FusedOp(
            name=f"online_{old_op.name}",
            reads=old_op.reads,
            computes=f"{old_op.computes}[{algorithm}]",
            flops_fn=old_op.flops_fn,  # same FLOPs (or slightly more)
            output=new_output,
            output_name=old_op.output_name,
            tiling=tiling,
            online_state=online_state,
            notes=f"Applied {algorithm}: output reduced from {old_op.output.shape} to {new_output.shape}",
        )

        ops[idx] = new_op
        return new_graph

    @staticmethod
    def replace_with_recompute(graph: ComputationGraph,
                                tensor_name: str,
                                recompute_in_op: str) -> ComputationGraph:
        """
        Change an intermediate tensor from "materialize to HBM" to "recompute when needed".

        The producer op's output changes to on_chip storage.
        The consumer op gets extra FLOPs (recomputation cost).
        IO savings = 2 × size(tensor) (no write + no read).
        """
        new_graph = graph.clone()

        # Find the producer op
        producer_idx = next(
            i for i, op in enumerate(new_graph.operations)
            if op.output_name == tensor_name
        )
        producer = new_graph.operations[producer_idx]

        # Change storage to on_chip
        producer.output = TensorSpec(
            shape=producer.output.shape,
            dtype=producer.output.dtype,
            storage="on_chip",
        )
        producer.notes += " [recomputed, not materialized]"

        # Find the consumer and add recomputation FLOPs
        for op in new_graph.operations:
            if op.name == recompute_in_op:
                old_fn = op.flops_fn
                producer_fn = producer.flops_fn
                op.flops_fn = lambda p, f1=old_fn, f2=producer_fn: f1(p) + f2(p)
                op.notes += f" [includes recomputation of {tensor_name}]"
                break

        return new_graph

    @staticmethod
    def apply_tiling(graph: ComputationGraph, op_name: str,
                     tile_dims: dict) -> ComputationGraph:
        """
        Apply tiling to an operation to fit working set in on-chip memory.
        """
        new_graph = graph.clone()
        idx = next(i for i, op in enumerate(new_graph.operations) if op.name == op_name)
        op = new_graph.operations[idx]
        op.tiling = TilingSpec(tiles=tile_dims)
        op.notes += f" [tiled: {tile_dims}]"
        return new_graph

    @staticmethod
    def fuse_and_online(graph: ComputationGraph,
                         ops_to_fuse: list[str],
                         online_algorithm: str,
                         output_name: str,
                         output_shape: tuple,
                         tiling: TilingSpec | None = None) -> ComputationGraph:
        """
        High-level action: fuse multiple ops AND apply online algorithm.

        This is the "Flash" pattern:
          1. Fuse producer + reduce into single pass
          2. Use online algorithm for the reduction
          3. Output is reduced (e.g., O(N) instead of O(NK))
        """
        new_graph = graph.clone()
        algo_spec = ONLINE_ALGORITHMS[online_algorithm]

        # Collect all ops to fuse
        fuse_indices = []
        for name in ops_to_fuse:
            idx = next(i for i, op in enumerate(new_graph.operations) if op.name == name)
            fuse_indices.append(idx)

        ops_to_merge = [new_graph.operations[i] for i in fuse_indices]

        # Combined reads = union of all reads, minus internal intermediates
        internal_names = {op.output_name for op in ops_to_merge}
        combined_reads = []
        seen = set()
        for op in ops_to_merge:
            for r in op.reads:
                if r not in internal_names and r not in seen:
                    combined_reads.append(r)
                    seen.add(r)

        # Combined FLOPs
        fns = [op.flops_fn for op in ops_to_merge]
        combined_flops_fn = lambda p, fns=fns: sum(f(p) for f in fns)

        online_state = OnlineStateSpec(
            variables=algo_spec["state_vars"],
            algorithm=online_algorithm,
            output_shape=output_shape,
        )

        new_output = TensorSpec(shape=output_shape, dtype="f32", storage="HBM")

        fused_name = "flash_" + "_".join(ops_to_fuse)
        fused = FusedOp(
            name=fused_name,
            reads=combined_reads,
            computes="+".join(op.computes for op in ops_to_merge) + f"[{online_algorithm}]",
            flops_fn=combined_flops_fn,
            output=new_output,
            output_name=output_name,
            tiling=tiling,
            online_state=online_state,
            notes=f"Flash fused: {ops_to_fuse}, online {online_algorithm}. "
                  f"Eliminated all intermediate materialization.",
        )

        # Remove fused ops, insert the new one
        new_ops = [op for i, op in enumerate(new_graph.operations) if i not in fuse_indices]
        new_ops.insert(min(fuse_indices), fused)
        new_graph.operations = new_ops

        # Clean up dangling references: remaining ops that read internal
        # intermediates of the fused group should have those refs removed
        for op in new_graph.operations:
            if op is not fused:
                op.reads = [r for r in op.reads if r not in internal_names]

        return new_graph

    @staticmethod
    def add_recompute_pass(graph: ComputationGraph,
                            pass_name: str,
                            reads: list[str],
                            computes: str,
                            flops_fn,
                            output: TensorSpec,
                            output_name: str,
                            notes: str = "") -> ComputationGraph:
        """
        Add a second pass that recomputes intermediate values and accumulates statistics.
        This is the Pass 2 of Flash-style algorithms.
        """
        new_graph = graph.clone()
        new_op = FusedOp(
            name=pass_name,
            reads=reads,
            computes=computes,
            flops_fn=flops_fn,
            output=output,
            output_name=output_name,
            notes=notes,
        )
        new_graph.operations.append(new_op)
        if output_name not in new_graph.outputs:
            new_graph.outputs.append(output_name)
        return new_graph

    @staticmethod
    def list_actions() -> list[dict]:
        """Return a description of all available actions for the agent."""
        return [
            {
                "name": "fuse_ops",
                "description": "Fuse two sequential ops, eliminating intermediate materialization",
                "io_effect": "Saves 2 × size(intermediate) bytes",
                "flop_effect": "No change (same total FLOPs)",
            },
            {
                "name": "apply_online_algorithm",
                "description": "Replace reduce with online/streaming algorithm",
                "io_effect": "Output shrinks from O(NK) to O(N)",
                "flop_effect": "Same or slightly more FLOPs",
                "algorithms": list(ONLINE_ALGORITHMS.keys()),
            },
            {
                "name": "replace_with_recompute",
                "description": "Recompute intermediate instead of materializing",
                "io_effect": "Saves 2 × size(intermediate) bytes",
                "flop_effect": "Doubles compute for the recomputed tensor",
            },
            {
                "name": "fuse_and_online",
                "description": "Fuse multiple ops + apply online algorithm (the 'Flash' pattern)",
                "io_effect": "Eliminates all intermediate materialization",
                "flop_effect": "Same total FLOPs for single pass",
            },
            {
                "name": "add_recompute_pass",
                "description": "Add a second pass that recomputes + accumulates stats",
                "io_effect": "Reads inputs again but writes only O(Kd) stats",
                "flop_effect": "Doubles total FLOPs",
            },
        ]
