"""
IO Complexity Calculator with Roofline Model.

Symbolically computes HBM read/write bytes for a ComputationGraph,
estimates wall-clock time using the roofline model, and identifies
optimization opportunities.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from .dsl import ComputationGraph, TensorSpec, Op


@dataclass
class HardwareSpec:
    """GPU hardware parameters for roofline modeling."""
    name: str
    hbm_bandwidth_gbps: float    # GB/s
    peak_flops_tflops: float     # TFLOPS (FP32)
    smem_per_sm_kb: float = 228  # KB
    l2_cache_mb: float = 50      # MB
    num_sms: int = 132

    @property
    def balance_point(self) -> float:
        """Arithmetic intensity (FLOP/Byte) where compute = memory time."""
        return self.peak_flops_tflops * 1e12 / (self.hbm_bandwidth_gbps * 1e9)

    def time_for_io_ms(self, io_bytes: int) -> float:
        return io_bytes / (self.hbm_bandwidth_gbps * 1e9) * 1000

    def time_for_flops_ms(self, flops: int) -> float:
        return flops / (self.peak_flops_tflops * 1e12) * 1000

    def roofline_time_ms(self, io_bytes: int, flops: int) -> float:
        return max(self.time_for_io_ms(io_bytes), self.time_for_flops_ms(flops))


# Pre-defined hardware specs
H200 = HardwareSpec("H200", hbm_bandwidth_gbps=4800, peak_flops_tflops=67, smem_per_sm_kb=228, l2_cache_mb=50, num_sms=132)
H100 = HardwareSpec("H100", hbm_bandwidth_gbps=3350, peak_flops_tflops=67, smem_per_sm_kb=228, l2_cache_mb=50, num_sms=132)
A100 = HardwareSpec("A100", hbm_bandwidth_gbps=2039, peak_flops_tflops=19.5, smem_per_sm_kb=164, l2_cache_mb=40, num_sms=108)


@dataclass
class OpIOStats:
    """IO statistics for a single operation."""
    name: str
    hbm_reads: int       # bytes
    hbm_writes: int      # bytes
    flops: int
    is_intermediate: bool
    storage: str
    is_fused: bool = False
    notes: str = ""

    @property
    def total_io(self) -> int:
        return self.hbm_reads + self.hbm_writes

    @property
    def arithmetic_intensity(self) -> float:
        if self.total_io == 0:
            return float('inf')
        return self.flops / self.total_io


@dataclass
class IOReport:
    """Complete IO analysis report for a computation graph."""
    graph_name: str
    params: dict
    op_stats: list[OpIOStats] = field(default_factory=list)
    hardware: HardwareSpec | None = None

    # Summary (computed)
    total_hbm_reads: int = 0
    total_hbm_writes: int = 0
    total_flops: int = 0
    estimated_time_ms: float = 0.0
    bottleneck: str = ""
    materialized_intermediates: list[str] = field(default_factory=list)
    optimization_hints: list[str] = field(default_factory=list)

    @property
    def total_io(self) -> int:
        return self.total_hbm_reads + self.total_hbm_writes

    @property
    def arithmetic_intensity(self) -> float:
        if self.total_io == 0:
            return float('inf')
        return self.total_flops / self.total_io

    def compute_summary(self, hw: HardwareSpec):
        self.hardware = hw
        self.total_hbm_reads = sum(op.hbm_reads for op in self.op_stats)
        self.total_hbm_writes = sum(op.hbm_writes for op in self.op_stats)
        self.total_flops = sum(op.flops for op in self.op_stats)

        io_time = hw.time_for_io_ms(self.total_io)
        compute_time = hw.time_for_flops_ms(self.total_flops)
        self.estimated_time_ms = max(io_time, compute_time)
        self.bottleneck = "memory-bound" if io_time >= compute_time else "compute-bound"

        # Identify materialized intermediates
        self.materialized_intermediates = [
            op.name for op in self.op_stats
            if op.is_intermediate and op.storage == "HBM"
        ]

        # Generate optimization hints
        self._generate_hints(hw)

    def _generate_hints(self, hw: HardwareSpec):
        self.optimization_hints = []

        for op in self.op_stats:
            if op.is_intermediate and op.storage == "HBM":
                self.optimization_hints.append(
                    f"[MATERIALIZE] '{op.name}' writes {op.hbm_writes/1e6:.1f} MB to HBM as intermediate. "
                    f"Consider fusing with consumer or using online algorithm to eliminate."
                )

            if op.arithmetic_intensity < hw.balance_point * 0.5:
                self.optimization_hints.append(
                    f"[MEMORY-BOUND] '{op.name}' has AI={op.arithmetic_intensity:.1f} FLOP/B "
                    f"(threshold={hw.balance_point:.1f}). IO dominates."
                )

        if self.bottleneck == "memory-bound":
            total_mat_bytes = sum(
                op.hbm_writes for op in self.op_stats
                if op.is_intermediate and op.storage == "HBM"
            )
            if total_mat_bytes > 0:
                saving_pct = total_mat_bytes / self.total_io * 100
                self.optimization_hints.append(
                    f"[OPPORTUNITY] Eliminating all intermediate materialization would save "
                    f"{total_mat_bytes/1e6:.1f} MB ({saving_pct:.0f}% of total IO)."
                )

    def display(self) -> str:
        lines = [
            f"╔══════════════════════════════════════════╗",
            f"║  IO Report: {self.graph_name:<28s}║",
            f"╠══════════════════════════════════════════╣",
            f"║  Params: {str({k:v for k,v in self.params.items() if isinstance(v, int)}):<31s}║",
            f"╠══════════════════════════════════════════╣",
        ]

        for op in self.op_stats:
            fused = " [FUSED]" if op.is_fused else ""
            mat = " *** MAT ***" if (op.is_intermediate and op.storage == "HBM") else ""
            lines.append(
                f"║  {op.name:<18s} R:{op.hbm_reads/1e6:>7.1f}MB "
                f"W:{op.hbm_writes/1e6:>7.1f}MB{fused}{mat}"
            )

        lines.extend([
            f"╠══════════════════════════════════════════╣",
            f"║  Total IO:    {self.total_io/1e6:>10.1f} MB              ║",
            f"║    Reads:     {self.total_hbm_reads/1e6:>10.1f} MB              ║",
            f"║    Writes:    {self.total_hbm_writes/1e6:>10.1f} MB              ║",
            f"║  Total FLOPs: {self.total_flops/1e9:>10.2f} GFLOPs          ║",
            f"║  Arith.Intensity: {self.arithmetic_intensity:>6.1f} FLOP/B          ║",
            f"║  Bottleneck:  {self.bottleneck:<26s}║",
        ])

        if self.hardware:
            lines.append(
                f"║  Est. time:   {self.estimated_time_ms:>10.3f} ms ({self.hardware.name:<8s})  ║"
            )

        lines.append(f"╠══════════════════════════════════════════╣")

        if self.materialized_intermediates:
            lines.append(f"║  ⚠ Materialized intermediates:           ║")
            for name in self.materialized_intermediates:
                lines.append(f"║    - {name:<35s}║")
        else:
            lines.append(f"║  ✓ No materialized intermediates          ║")

        if self.optimization_hints:
            lines.append(f"╠══════════════════════════════════════════╣")
            lines.append(f"║  Optimization Hints:                     ║")
            for hint in self.optimization_hints:
                # Wrap long hints
                for i in range(0, len(hint), 40):
                    chunk = hint[i:i+40]
                    lines.append(f"║    {chunk:<37s}║")

        lines.append(f"╚══════════════════════════════════════════╝")
        return "\n".join(lines)


@dataclass
class ComparisonReport:
    """Comparison of baseline vs optimized computation graph."""
    baseline_name: str
    optimized_name: str
    baseline: IOReport
    optimized: IOReport

    @property
    def io_reduction(self) -> float:
        return 1 - self.optimized.total_io / self.baseline.total_io

    @property
    def flop_change(self) -> float:
        return self.optimized.total_flops / self.baseline.total_flops - 1

    @property
    def estimated_speedup(self) -> float:
        if self.optimized.estimated_time_ms == 0:
            return float('inf')
        return self.baseline.estimated_time_ms / self.optimized.estimated_time_ms

    @property
    def eliminated_intermediates(self) -> list[str]:
        return [m for m in self.baseline.materialized_intermediates
                if m not in self.optimized.materialized_intermediates]

    def display(self) -> str:
        lines = [
            f"",
            f"┌────────────────────────────────────────────────────┐",
            f"│  Comparison: {self.baseline_name} → {self.optimized_name}",
            f"├────────────────────────────────────────────────────┤",
            f"│  IO Reduction:      {self.io_reduction*100:>6.1f}%                        │",
            f"│  FLOP Change:       {self.flop_change*100:>+6.1f}%                        │",
            f"│  Estimated Speedup: {self.estimated_speedup:>6.2f}×                        │",
            f"├────────────────────────────────────────────────────┤",
            f"│  IO:   {self.baseline.total_io/1e6:>8.1f} MB → {self.optimized.total_io/1e6:>8.1f} MB           │",
            f"│  FLOPs:{self.baseline.total_flops/1e9:>8.2f} GF → {self.optimized.total_flops/1e9:>8.2f} GF           │",
            f"│  Time: {self.baseline.estimated_time_ms:>8.3f} ms → {self.optimized.estimated_time_ms:>8.3f} ms           │",
        ]

        if self.eliminated_intermediates:
            lines.append(f"├────────────────────────────────────────────────────┤")
            lines.append(f"│  Eliminated materialization:                       │")
            for name in self.eliminated_intermediates:
                lines.append(f"│    ✓ {name:<44s}│")

        verdict = "BETTER" if self.estimated_speedup > 1.0 else "WORSE"
        lines.extend([
            f"├────────────────────────────────────────────────────┤",
            f"│  Verdict: {verdict:<40s}│",
            f"└────────────────────────────────────────────────────┘",
        ])
        return "\n".join(lines)


class IOCalculator:
    """
    Symbolic IO complexity calculator.

    Analyzes a ComputationGraph and produces an IOReport
    with HBM byte counts, FLOP counts, and roofline-based
    performance estimates.
    """

    def __init__(self, hardware: HardwareSpec = H200):
        self.hw = hardware

    def analyze(self, graph: ComputationGraph, params: dict) -> IOReport:
        """Analyze a computation graph and produce an IO report."""
        report = IOReport(graph_name=graph.name, params=params)

        # Track which tensors are "live" (available in HBM)
        available = set(graph.inputs.keys())

        for op in graph.operations:
            # Calculate reads: sum of sizes of all read tensors
            total_reads = 0
            for read_name in op.reads:
                tensor = graph.get_tensor(read_name)
                if tensor.storage == "HBM" or read_name in available:
                    if op.is_fused and op.tiling:
                        # Fused op with tiling: reads input once per pass
                        # (streaming through tiles, each tile loaded once)
                        total_reads += tensor.size_bytes(params)
                    else:
                        total_reads += tensor.size_bytes(params)

            # Calculate writes
            total_writes = 0
            if op.output.storage == "HBM":
                total_writes = op.output.size_bytes(params)

            # Is this an intermediate?
            is_intermediate = op.output_name not in graph.outputs

            flops = op.flops(params)

            report.op_stats.append(OpIOStats(
                name=op.name,
                hbm_reads=total_reads,
                hbm_writes=total_writes,
                flops=flops,
                is_intermediate=is_intermediate,
                storage=op.output.storage,
                is_fused=op.is_fused,
                notes=op.notes,
            ))

            # Mark output as available for subsequent ops
            if op.output.storage == "HBM":
                available.add(op.output_name)

        report.compute_summary(self.hw)
        return report

    def compare(self, baseline: ComputationGraph, optimized: ComputationGraph,
                params: dict) -> ComparisonReport:
        """Compare IO efficiency of baseline vs optimized graph."""
        r1 = self.analyze(baseline, params)
        r2 = self.analyze(optimized, params)
        return ComparisonReport(
            baseline_name=baseline.name,
            optimized_name=optimized.name,
            baseline=r1,
            optimized=r2,
        )
