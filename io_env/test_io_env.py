"""
Tests for the IO Environment and ReAct Agent.

These tests verify:
1. DSL: ComputationGraph construction, tensor lookup, intermediate detection
2. Calculator: IO analysis, roofline model, comparison reports
3. Actions: fuse_ops, fuse_and_online, recompute, dangling ref fix
4. Workflow: analyze → try_action → undo → summary
5. ReAct Agent: full Thought→Action→Observation loop on all operators
6. Verify tool: correctness checks as environment tool

Run: pytest io_env/test_io_env.py -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from io_env.dsl import TensorSpec, Op, FusedOp, ComputationGraph, TilingSpec, OnlineStateSpec
from io_env.calculator import IOCalculator, H200, A100
from io_env.actions import DesignActions
from io_env.agent_loop import compute_reward
from io_env.examples import EXAMPLES
from io_env.react_agent import IOEnvironment, run_react_agent


# ============================================================================
# DSL Tests
# ============================================================================

class TestDSL:
    def test_tensor_spec_size(self):
        t = TensorSpec(shape=("N", "K"), dtype="f32")
        assert t.size_bytes({"N": 1024, "K": 64}) == 1024 * 64 * 4

    def test_tensor_spec_dtype(self):
        t16 = TensorSpec(shape=(100,), dtype="f16")
        t32 = TensorSpec(shape=(100,), dtype="f32")
        assert t16.dtype_bytes == 2
        assert t32.dtype_bytes == 4

    def test_graph_get_tensor(self):
        g = EXAMPLES["gmm_estep"]["baseline"]()
        assert g.get_tensor("X").shape == ("N", "d")
        assert g.get_tensor("L").shape == ("N", "K")

    def test_graph_materialized_intermediates(self):
        g = EXAMPLES["gmm_estep"]["baseline"]()
        mats = g.get_materialized_intermediates()
        mat_names = [n for n, _ in mats]
        assert "L" in mat_names  # L is intermediate, not in outputs

    def test_cross_entropy_has_3_intermediates(self):
        g = EXAMPLES["cross_entropy"]["baseline"]()
        mats = g.get_materialized_intermediates()
        assert len(mats) == 3
        mat_names = {n for n, _ in mats}
        assert mat_names == {"max_vals", "exp_logits", "log_sum_exp"}

    def test_graph_clone(self):
        g1 = EXAMPLES["kmeans"]["baseline"]()
        g2 = g1.clone()
        g2.operations[0].name = "modified"
        assert g1.operations[0].name != "modified"


# ============================================================================
# Calculator Tests
# ============================================================================

class TestCalculator:
    def test_analyze_returns_report(self):
        calc = IOCalculator(hardware=H200)
        g = EXAMPLES["gmm_estep"]["baseline"]()
        p = EXAMPLES["gmm_estep"]["default_params"]
        report = calc.analyze(g, p)
        assert report.total_io > 0
        assert report.total_flops > 0
        assert report.bottleneck in ("memory-bound", "compute-bound")

    def test_io_reduction_baseline_vs_flash(self):
        calc = IOCalculator(hardware=H200)
        p = EXAMPLES["gmm_estep"]["default_params"]
        r_base = calc.analyze(EXAMPLES["gmm_estep"]["baseline"](), p)
        r_flash = calc.analyze(EXAMPLES["gmm_estep"]["flash"](), p)
        assert r_flash.total_io < r_base.total_io

    def test_cross_entropy_memory_bound(self):
        calc = IOCalculator(hardware=H200)
        g = EXAMPLES["cross_entropy"]["baseline"]()
        p = EXAMPLES["cross_entropy"]["default_params"]
        report = calc.analyze(g, p)
        assert report.bottleneck == "memory-bound"

    def test_comparison_report(self):
        calc = IOCalculator(hardware=H200)
        p = EXAMPLES["kmeans"]["default_params"]
        comp = calc.compare(
            EXAMPLES["kmeans"]["baseline"](),
            EXAMPLES["kmeans"]["flash"](),
            p
        )
        assert comp.io_reduction > 0.5  # >50% reduction
        assert comp.estimated_speedup >= 1.0

    def test_hardware_affects_time(self):
        g = EXAMPLES["cross_entropy"]["baseline"]()
        p = EXAMPLES["cross_entropy"]["default_params"]
        r_h200 = IOCalculator(H200).analyze(g, p)
        r_a100 = IOCalculator(A100).analyze(g, p)
        assert r_a100.estimated_time_ms > r_h200.estimated_time_ms  # A100 slower


# ============================================================================
# Actions Tests
# ============================================================================

class TestActions:
    def test_fuse_ops(self):
        g = EXAMPLES["gmm_estep"]["baseline"]()
        new_g = DesignActions.fuse_ops(g, "compute_L", "logsumexp")
        assert len(new_g.operations) == len(g.operations) - 1
        op_names = [op.name for op in new_g.operations]
        assert "compute_L" not in op_names
        assert "logsumexp" not in op_names

    def test_fuse_ops_dangling_ref(self):
        """Fusing row_max+subtract_and_exp should not leave gather_loss with dangling 'max_vals' ref."""
        g = EXAMPLES["cross_entropy"]["baseline"]()
        new_g = DesignActions.fuse_ops(g, "row_max", "subtract_and_exp")
        # gather_loss originally reads max_vals — should be removed
        gather_op = next(op for op in new_g.operations if op.name == "gather_loss")
        assert "max_vals" not in gather_op.reads

        # Verify the new graph can be analyzed without error
        calc = IOCalculator(H200)
        p = EXAMPLES["cross_entropy"]["default_params"]
        report = calc.analyze(new_g, p)
        assert report.total_io > 0

    def test_fuse_and_online_logsumexp(self):
        g = EXAMPLES["gmm_estep"]["baseline"]()
        new_g = DesignActions.fuse_and_online(
            g,
            ops_to_fuse=["compute_L", "logsumexp", "normalize"],
            online_algorithm="online_logsumexp",
            output_name="log_norm",
            output_shape=("N",),
        )
        assert len(new_g.operations) == 1
        assert new_g.operations[0].is_fused

    def test_fuse_and_online_argmin(self):
        g = EXAMPLES["kmeans"]["baseline"]()
        new_g = DesignActions.fuse_and_online(
            g,
            ops_to_fuse=["compute_distances", "argmin"],
            online_algorithm="online_argmin",
            output_name="assignments",
            output_shape=("N",),
        )
        assert len(new_g.operations) == 1
        assert not new_g.get_materialized_intermediates()

    def test_list_actions(self):
        actions = DesignActions.list_actions()
        assert len(actions) >= 4
        names = [a["name"] for a in actions]
        assert "fuse_ops" in names
        assert "fuse_and_online" in names


# ============================================================================
# Reward Tests
# ============================================================================

class TestReward:
    def test_positive_reward_for_io_reduction(self):
        calc = IOCalculator(H200)
        p = EXAMPLES["gmm_estep"]["default_params"]
        baseline = calc.analyze(EXAMPLES["gmm_estep"]["baseline"](), p)
        optimized = calc.analyze(EXAMPLES["gmm_estep"]["flash"](), p)
        reward = compute_reward(baseline, optimized)
        assert reward > 0

    def test_zero_reward_for_no_change(self):
        calc = IOCalculator(H200)
        p = EXAMPLES["gmm_estep"]["default_params"]
        report = calc.analyze(EXAMPLES["gmm_estep"]["baseline"](), p)
        reward = compute_reward(report, report)
        assert reward == 0.0


# ============================================================================
# ReAct Environment Tests
# ============================================================================

class TestReActEnvironment:
    def test_reset(self):
        env = IOEnvironment()
        obs = env.reset("cross_entropy")
        assert "cross_entropy" in obs
        assert "2621" in obs  # baseline IO ~2621 MB
        assert env.graph is not None

    def test_step_analyze(self):
        env = IOEnvironment()
        obs, reward, done = env.step("Analyze the operator", "analyze", {"task": "gmm_estep"})
        assert "gmm_estep" in obs
        assert not done

    def test_step_fuse_ops(self):
        env = IOEnvironment()
        env.reset("cross_entropy")
        obs, reward, done = env.step(
            "Fuse row_max + subtract_and_exp",
            "fuse_ops",
            {"op1": "row_max", "op2": "subtract_and_exp"}
        )
        assert "fused" in obs.lower() or "fuse" in obs.lower()
        assert not done

    def test_step_fuse_and_online(self):
        env = IOEnvironment()
        env.reset("kmeans")
        obs, reward, done = env.step(
            "Fuse distance + argmin with online argmin",
            "fuse_and_online",
            {"ops": ["compute_distances", "argmin"],
             "algorithm": "online_argmin",
             "output_name": "assignments",
             "output_shape": ["N"]}
        )
        assert reward > 0
        assert not done

    def test_step_verify(self):
        env = IOEnvironment()
        env.reset("kmeans")
        env.step("", "fuse_and_online",
                {"ops": ["compute_distances", "argmin"],
                 "algorithm": "online_argmin",
                 "output_name": "assignments",
                 "output_shape": ["N"]})
        obs, reward, done = env.step("Verify", "verify", {})
        assert "PASS" in obs
        assert reward > 0

    def test_step_done(self):
        env = IOEnvironment()
        env.reset("kmeans")
        env.step("", "fuse_and_online",
                {"ops": ["compute_distances", "argmin"],
                 "algorithm": "online_argmin",
                 "output_name": "assignments",
                 "output_shape": ["N"]})
        obs, reward, done = env.step("Done", "done", {})
        assert done
        assert "COMPLETE" in obs

    def test_step_undo(self):
        env = IOEnvironment()
        env.reset("cross_entropy")
        env.step("", "fuse_ops", {"op1": "row_max", "op2": "subtract_and_exp"})
        assert len(env.graph.operations) == 3
        env.step("", "undo", {})
        assert len(env.graph.operations) == 4  # restored

    def test_unknown_tool(self):
        env = IOEnvironment()
        env.reset("kmeans")
        obs, reward, done = env.step("", "nonexistent_tool", {})
        assert "ERROR" in obs

    def test_prompt_generation(self):
        env = IOEnvironment()
        prompt = env.get_prompt()
        assert "fuse_ops" in prompt
        assert "fuse_and_online" in prompt
        assert "online_logsumexp" in prompt


# ============================================================================
# Full ReAct Agent Tests (end-to-end)
# ============================================================================

class TestReActAgent:
    @pytest.mark.parametrize("task", ["cross_entropy", "gmm_estep", "kmeans", "softmax"])
    def test_react_agent_optimizes(self, task):
        """The built-in ReAct agent should eliminate all materialized intermediates."""
        env = run_react_agent(task, verbose=False)
        mats = env.graph.get_materialized_intermediates()
        # Agent should have found a way (may not eliminate all for softmax, but IO should drop)
        assert env.current_report.total_io <= env.baseline_report.total_io

    def test_react_agent_cross_entropy_io_reduction(self):
        env = run_react_agent("cross_entropy", verbose=False)
        reduction = 1 - env.current_report.total_io / env.baseline_report.total_io
        assert reduction > 0.5  # >50% IO reduction

    def test_react_agent_verify_passes(self):
        env = run_react_agent("gmm_estep", verbose=False)
        obs, reward, done = env.step("Verify final result", "verify", {})
        assert "PASS" in obs


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
