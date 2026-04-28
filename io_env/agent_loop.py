"""
Agent Loop: LLM-driven IO-aware operator optimization.

The agent receives an IO analysis report, proposes optimization actions,
and iteratively improves the computation graph guided by IO reduction rewards.
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from .dsl import ComputationGraph, TilingSpec
from .calculator import IOCalculator, IOReport, ComparisonReport, HardwareSpec, H200
from .actions import DesignActions, ONLINE_ALGORITHMS


@dataclass
class StepRecord:
    """Record of one agent optimization step."""
    step: int
    action: str
    action_details: dict
    io_before: int
    io_after: int
    io_reduction_pct: float
    flop_change_pct: float
    estimated_speedup: float
    graph_name: str
    reward: float


@dataclass
class AgentSession:
    """Complete record of an agent optimization session."""
    task: str
    hardware: str
    params: dict
    steps: list[StepRecord] = field(default_factory=list)
    baseline_io: int = 0
    final_io: int = 0
    total_reward: float = 0.0


def compute_reward(baseline: IOReport, optimized: IOReport) -> float:
    """Multi-dimensional reward signal for the agent."""
    reward = 0.0

    # R1: IO reduction (primary signal)
    io_reduction = 1 - optimized.total_io / baseline.total_io
    reward += 5.0 * max(io_reduction, 0)

    # R2: Estimated speedup (log scale)
    if optimized.estimated_time_ms > 0:
        speedup = baseline.estimated_time_ms / optimized.estimated_time_ms
        reward += 3.0 * math.log2(max(speedup, 0.5))

    # R3: Eliminated intermediates
    eliminated = len(baseline.materialized_intermediates) - len(optimized.materialized_intermediates)
    reward += 0.5 * eliminated

    # R4: FLOP penalty (avoid excessive recomputation)
    flop_ratio = optimized.total_flops / max(baseline.total_flops, 1)
    if flop_ratio > 4.0:
        reward -= 1.0 * (flop_ratio - 4.0)

    return reward


class RuleBasedAgent:
    """
    A rule-based agent that systematically applies IO optimizations.
    
    This serves as a deterministic baseline that demonstrates the
    environment works. A real LLM agent would replace this with
    learned/prompted decision-making.
    """

    def __init__(self, calculator: IOCalculator):
        self.calc = calculator

    def optimize(self, graph: ComputationGraph, params: dict,
                 verbose: bool = True) -> tuple[ComputationGraph, AgentSession]:
        """Run the agent optimization loop."""
        session = AgentSession(
            task=graph.name,
            hardware=self.calc.hw.name,
            params=params,
        )

        baseline_report = self.calc.analyze(graph, params)
        session.baseline_io = baseline_report.total_io
        current_graph = graph
        current_report = baseline_report
        step = 0

        if verbose:
            print("=" * 60)
            print(f"🤖 Agent starting optimization of: {graph.name}")
            print(f"   Hardware: {self.calc.hw.name}")
            print(f"   Params: { {k:v for k,v in params.items() if isinstance(v, int)} }")
            print("=" * 60)
            print()
            print("📊 Baseline analysis:")
            print(current_report.display())
            print()

        # Strategy 1: Look for materialized intermediates to eliminate
        mats = current_graph.get_materialized_intermediates()
        if mats and verbose:
            print(f"🔍 Found {len(mats)} materialized intermediate(s):")
            for name, spec in mats:
                print(f"   - {name}: {spec} ({spec.size_bytes(params)/1e6:.1f} MB)")
            print()

        # Strategy 2: Try to fuse ALL ops when multiple intermediates exist
        mats = current_graph.get_materialized_intermediates()
        if len(mats) > 1:
            # Multiple intermediates — try fusing all ops at once
            all_op_names = [op.name for op in current_graph.operations]
            # Find the final output op
            final_output = current_graph.outputs[0]
            final_op = next(op for op in current_graph.operations
                           if op.output_name == final_output)

            can_online = self._find_online_algorithm_chain(current_graph, params)
            if can_online:
                algo_name, output_shape = can_online
                if verbose:
                    print(f"💡 Step {step}: Apply fuse_and_online (full chain)")
                    print(f"   Fusing all {len(all_op_names)} ops: {all_op_names}")
                    print(f"   Online algorithm: {algo_name}")
                    print(f"   Output shape: → {output_shape}")

                try:
                    new_graph = DesignActions.fuse_and_online(
                        current_graph,
                        ops_to_fuse=all_op_names,
                        online_algorithm=algo_name,
                        output_name=final_output,
                        output_shape=output_shape,
                        tiling=TilingSpec(tiles={"N": "BN", "K": "BK"}) if "K" in params else None,
                    )

                    new_report = self.calc.analyze(new_graph, params)
                    reward = compute_reward(baseline_report, new_report)
                    comparison = ComparisonReport(
                        baseline_name=current_graph.name,
                        optimized_name=new_graph.name,
                        baseline=current_report,
                        optimized=new_report,
                    )
                    record = StepRecord(
                        step=step,
                        action=f"fuse_and_online({algo_name})",
                        action_details={"fused_ops": all_op_names, "eliminated": [n for n, _ in mats]},
                        io_before=current_report.total_io,
                        io_after=new_report.total_io,
                        io_reduction_pct=(1 - new_report.total_io / current_report.total_io) * 100,
                        flop_change_pct=(new_report.total_flops / current_report.total_flops - 1) * 100,
                        estimated_speedup=current_report.estimated_time_ms / max(new_report.estimated_time_ms, 1e-9),
                        graph_name=new_graph.name,
                        reward=reward,
                    )
                    session.steps.append(record)
                    session.total_reward += reward

                    if verbose:
                        print(comparison.display())
                        print(f"   Reward: {reward:.2f}")
                        print()

                    current_graph = new_graph
                    current_report = new_report
                    step += 1
                    # Skip the while loop below since we fused everything
                    mats = []
                except Exception as e:
                    if verbose:
                        print(f"   ⚠ Failed: {e}, falling back to incremental")

        # Strategy 3: Incremental — fuse ops that produce/consume intermediates
        mats_remaining = current_graph.get_materialized_intermediates()
        while mats_remaining:
            # Target the largest materialized intermediate
            mats_sorted = sorted(mats_remaining, key=lambda m: m[1].num_elements(params), reverse=True)
            mat_name, mat_spec = mats_sorted[0]

            # Find producer and all consumers
            producer = None
            consumers = []
            for op in current_graph.operations:
                if op.output_name == mat_name:
                    producer = op
                for r in op.reads:
                    if r == mat_name:
                        consumers.append(op)
                        break  # avoid duplicate per op

            if producer is None or not consumers:
                break

            # Check if removing this intermediate would leave dangling refs from other ops
            all_refs = set()
            for op in current_graph.operations:
                if op not in consumers and op != producer:
                    all_refs.update(op.reads)

            if mat_name in all_refs:
                # Other ops also read this intermediate — must fuse them all
                extra_consumers = [op for op in current_graph.operations
                                   if mat_name in op.reads and op not in consumers]
                consumers.extend(extra_consumers)

            # Decide: can we use an online algorithm?
            can_online = self._find_online_algorithm(producer, consumers, params)

            if can_online:
                algo_name, output_shape = can_online
                if verbose:
                    print(f"💡 Step {step}: Apply fuse_and_online")
                    print(f"   Fusing: {[producer.name] + [c.name for c in consumers]}")
                    print(f"   Online algorithm: {algo_name}")
                    print(f"   Output shape: {mat_spec.shape} → {output_shape}")

                ops_to_fuse = [producer.name] + [c.name for c in consumers]
                output_name = consumers[-1].output_name

                try:
                    new_graph = DesignActions.fuse_and_online(
                        current_graph,
                        ops_to_fuse=ops_to_fuse,
                        online_algorithm=algo_name,
                        output_name=output_name,
                        output_shape=output_shape,
                        tiling=TilingSpec(tiles={"N": "BN", "K": "BK"}) if "K" in params else None,
                    )
                except Exception as e:
                    if verbose:
                        print(f"   ⚠ Failed: {e}")
                    break

            else:
                # Fallback: simple fusion
                if verbose:
                    print(f"💡 Step {step}: Apply fuse_ops({producer.name}, {consumers[0].name})")

                try:
                    new_graph = DesignActions.fuse_ops(
                        current_graph, producer.name, consumers[0].name)
                except Exception as e:
                    if verbose:
                        print(f"   ⚠ Failed: {e}")
                    break

            # Evaluate
            new_report = self.calc.analyze(new_graph, params)
            reward = compute_reward(baseline_report, new_report)

            comparison = ComparisonReport(
                baseline_name=current_graph.name,
                optimized_name=new_graph.name,
                baseline=current_report,
                optimized=new_report,
            )

            record = StepRecord(
                step=step,
                action=can_online[0] if can_online else "fuse_ops",
                action_details={"eliminated": mat_name},
                io_before=current_report.total_io,
                io_after=new_report.total_io,
                io_reduction_pct=(1 - new_report.total_io / current_report.total_io) * 100,
                flop_change_pct=(new_report.total_flops / current_report.total_flops - 1) * 100,
                estimated_speedup=current_report.estimated_time_ms / max(new_report.estimated_time_ms, 1e-9),
                graph_name=new_graph.name,
                reward=reward,
            )
            session.steps.append(record)
            session.total_reward += reward

            if verbose:
                print(comparison.display())
                print(f"   Reward: {reward:.2f}")
                print()

            current_graph = new_graph
            current_report = new_report
            step += 1
            mats_remaining = current_graph.get_materialized_intermediates()

        if not current_graph.get_materialized_intermediates():
            if verbose:
                print("✅ No more materialized intermediates to eliminate.")
        if "gmm" in graph.name.lower() and "gamma" not in [op.output_name for op in current_graph.operations]:
            if verbose:
                print(f"💡 Step {step}: Add recompute pass for M-step statistics")

            new_graph = DesignActions.add_recompute_pass(
                current_graph,
                pass_name="flash_accumulate_stats",
                reads=["X", "mu", "var", "log_pi", "log_norm"],
                computes="recompute_gamma+accumulate_stats",
                flops_fn=lambda p: p["N"] * p["K"] * (6 * p["d"] + 5),
                output=TensorSpec(shape=("K", "d"), dtype="f32", storage="HBM"),
                output_name="sufficient_stats",
                notes="Pass 2: recompute γ on-chip, accumulate n_k, s_k, sq_k",
            )

            from .dsl import TensorSpec  # re-import after copy
            new_report = self.calc.analyze(new_graph, params)
            reward = compute_reward(baseline_report, new_report)

            record = StepRecord(
                step=step,
                action="add_recompute_pass",
                action_details={"pass": "accumulate_stats"},
                io_before=current_report.total_io,
                io_after=new_report.total_io,
                io_reduction_pct=(1 - new_report.total_io / baseline_report.total_io) * 100,
                flop_change_pct=(new_report.total_flops / baseline_report.total_flops - 1) * 100,
                estimated_speedup=baseline_report.estimated_time_ms / max(new_report.estimated_time_ms, 1e-9),
                graph_name=new_graph.name,
                reward=reward,
            )
            session.steps.append(record)
            session.total_reward += reward

            if verbose:
                comparison = ComparisonReport(
                    baseline_name=graph.name,
                    optimized_name=new_graph.name,
                    baseline=baseline_report,
                    optimized=new_report,
                )
                print(comparison.display())
                print(f"   Reward: {reward:.2f}")
                print()

            current_graph = new_graph
            current_report = new_report

        session.final_io = current_report.total_io

        if verbose:
            self._print_summary(session, baseline_report, current_report)

        return current_graph, session

    def _find_online_algorithm(self, producer, consumers, params):
        """Check if an online algorithm can replace the producer+consumer chain."""
        combined_computes = producer.computes + "+" + "+".join(c.computes for c in consumers)

        # Check for logsumexp pattern
        if "log_likelihood" in producer.computes and any("reduce" in c.computes or "logsumexp" in c.computes for c in consumers):
            return ("online_logsumexp", ("N",))

        # Check for argmin pattern
        if ("distance" in producer.computes or "l2" in producer.computes) and any("argmin" in c.computes for c in consumers):
            return ("online_argmin", ("N",))

        # Check for softmax pattern
        if "matmul" in producer.computes and any("softmax" in c.computes for c in consumers):
            return ("online_softmax", ("N", "d"))

        # Check for general reduce pattern
        if any("reduce" in c.computes for c in consumers):
            return ("online_logsumexp", ("N",))

        return None

    def _find_online_algorithm_chain(self, graph, params):
        """Check if the entire op chain can be replaced by an online algorithm."""
        all_computes = " ".join(op.computes for op in graph.operations)

        # Cross-entropy pattern: max + exp + sum + log + gather
        if ("reduce_max" in all_computes and "exp" in all_computes and
            ("reduce_sum" in all_computes or "sum" in all_computes)):
            return ("online_logsumexp", ("N",))

        # General logsumexp pattern
        if "logsumexp" in all_computes or ("max" in all_computes and "exp" in all_computes):
            return ("online_logsumexp", ("N",))

        return None

    def _print_summary(self, session, baseline, final):
        print("=" * 60)
        print(f"📋 Optimization Summary")
        print("=" * 60)
        print(f"  Task:     {session.task}")
        print(f"  Hardware: {session.hardware}")
        print(f"  Steps:    {len(session.steps)}")
        print()
        print(f"  Baseline IO:  {session.baseline_io / 1e6:>10.1f} MB")
        print(f"  Final IO:     {session.final_io / 1e6:>10.1f} MB")
        print(f"  IO Reduction: {(1 - session.final_io / session.baseline_io) * 100:.1f}%")
        print()
        print(f"  Baseline time: {baseline.estimated_time_ms:.3f} ms")
        print(f"  Final time:    {final.estimated_time_ms:.3f} ms")
        print(f"  Speedup:       {baseline.estimated_time_ms / max(final.estimated_time_ms, 1e-9):.2f}×")
        print()
        print(f"  Total Reward:  {session.total_reward:.2f}")
        print()

        print("  Step-by-step:")
        for s in session.steps:
            print(f"    Step {s.step}: {s.action}")
            print(f"      IO: {s.io_before/1e6:.1f} → {s.io_after/1e6:.1f} MB "
                  f"({s.io_reduction_pct:+.1f}%)")
            print(f"      FLOPs: {s.flop_change_pct:+.1f}%, Reward: {s.reward:.2f}")
        print("=" * 60)
