#!/usr/bin/env python3
"""
IO Environment Demo: Agent discovers Flash-style optimizations.

Runs the rule-based agent on multiple operator examples and shows
how the IO Calculator guides optimization decisions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from io_env.calculator import IOCalculator, H200, H100, A100
from io_env.actions import DesignActions
from io_env.agent_loop import RuleBasedAgent
from io_env.examples import EXAMPLES


def demo_io_analysis():
    """Demo 1: Pure IO analysis of baseline vs flash, no agent."""
    print("\n" + "█" * 60)
    print("█  Demo 1: IO Complexity Analysis                        █")
    print("█" * 60 + "\n")

    calc = IOCalculator(hardware=H200)

    for name, example in EXAMPLES.items():
        params = example["default_params"]
        baseline = example["baseline"]()
        flash = example["flash"]()

        print(f"\n{'─' * 50}")
        print(f"  {name.upper()}")
        print(f"{'─' * 50}")

        # Baseline analysis
        report_baseline = calc.analyze(baseline, params)
        print("\n[Baseline]")
        print(report_baseline.display())

        # Flash analysis
        report_flash = calc.analyze(flash, params)
        print("\n[Flash]")
        print(report_flash.display())

        # Comparison
        comparison = calc.compare(baseline, flash, params)
        print(comparison.display())


def demo_agent_optimization():
    """Demo 2: Agent discovers optimizations step by step."""
    print("\n" + "█" * 60)
    print("█  Demo 2: Agent Optimization Loop                      █")
    print("█" * 60 + "\n")

    calc = IOCalculator(hardware=H200)
    agent = RuleBasedAgent(calc)

    # Test on each example
    for name, example in EXAMPLES.items():
        if name in ("gmm_em_fused",):  # skip derived examples
            continue

        params = example["default_params"]
        baseline = example["baseline"]()

        print(f"\n{'━' * 60}")
        print(f"  Task: {name.upper()}")
        print(f"{'━' * 60}\n")

        optimized_graph, session = agent.optimize(baseline, params, verbose=True)
        print()


def demo_scaling():
    """Demo 3: Show how IO savings scale with problem size."""
    print("\n" + "█" * 60)
    print("█  Demo 3: Scaling Analysis                              █")
    print("█" * 60 + "\n")

    calc = IOCalculator(hardware=H200)

    example = EXAMPLES["gmm_estep"]
    baseline_fn = example["baseline"]
    flash_fn = example["flash"]

    print(f"{'N':>10s} {'K':>6s} {'d':>4s} │ {'Baseline IO':>12s} {'Flash IO':>12s} {'Reduction':>10s} {'Speedup':>8s}")
    print("─" * 70)

    configs = [
        {"N": 4096,   "K": 64,   "d": 32,  "BN": 64, "BK": 8},
        {"N": 16384,  "K": 256,  "d": 64,  "BN": 64, "BK": 8},
        {"N": 65536,  "K": 1024, "d": 128, "BN": 64, "BK": 8},
        {"N": 262144, "K": 2048, "d": 128, "BN": 64, "BK": 8},
        {"N": 1048576,"K": 4096, "d": 128, "BN": 64, "BK": 8},
        {"N": 1048576,"K": 8192, "d": 128, "BN": 64, "BK": 8},
    ]

    for params in configs:
        baseline = baseline_fn()
        flash = flash_fn()

        r_base = calc.analyze(baseline, params)
        r_flash = calc.analyze(flash, params)

        reduction = (1 - r_flash.total_io / r_base.total_io) * 100
        speedup = r_base.estimated_time_ms / max(r_flash.estimated_time_ms, 1e-9)

        print(f"{params['N']:>10,d} {params['K']:>6,d} {params['d']:>4d} │ "
              f"{r_base.total_io/1e9:>10.2f} GB {r_flash.total_io/1e9:>10.2f} GB "
              f"{reduction:>8.1f}% {speedup:>7.1f}×")


def demo_hardware_comparison():
    """Demo 4: Same workload on different GPUs."""
    print("\n" + "█" * 60)
    print("█  Demo 4: Hardware Comparison                           █")
    print("█" * 60 + "\n")

    params = {"N": 65536, "K": 1024, "d": 128, "BN": 64, "BK": 8}

    example = EXAMPLES["gmm_estep"]
    baseline = example["baseline"]()
    flash = example["flash"]()

    print(f"{'GPU':>8s} │ {'Baseline':>10s} {'Flash':>10s} {'Speedup':>8s} │ {'Balance Pt':>10s} {'Bottleneck':>12s}")
    print("─" * 70)

    for hw in [A100, H100, H200]:
        calc = IOCalculator(hardware=hw)
        r_base = calc.analyze(baseline, params)
        r_flash = calc.analyze(flash, params)
        speedup = r_base.estimated_time_ms / max(r_flash.estimated_time_ms, 1e-9)

        print(f"{hw.name:>8s} │ {r_base.estimated_time_ms:>8.3f}ms {r_flash.estimated_time_ms:>8.3f}ms "
              f"{speedup:>7.1f}× │ {hw.balance_point:>8.1f} F/B {r_base.bottleneck:>12s}")


def demo_action_space():
    """Demo 5: Show available design actions for the agent."""
    print("\n" + "█" * 60)
    print("█  Demo 5: Agent Action Space                            █")
    print("█" * 60 + "\n")

    actions = DesignActions.list_actions()
    for i, action in enumerate(actions):
        print(f"  Action {i+1}: {action['name']}")
        print(f"    {action['description']}")
        print(f"    IO effect:   {action['io_effect']}")
        print(f"    FLOP effect: {action['flop_effect']}")
        if 'algorithms' in action:
            print(f"    Algorithms:  {action['algorithms']}")
        print()


if __name__ == "__main__":
    demo_action_space()
    demo_io_analysis()
    demo_agent_optimization()
    demo_scaling()
    demo_hardware_comparison()
