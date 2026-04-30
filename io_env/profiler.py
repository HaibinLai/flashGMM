#!/usr/bin/env python3
"""
Runtime Profiling Module for the IO-Aware Agent.

Provides data-driven kernel diagnostics using:
  - torch.profiler (default, zero install cost)
  - ncu (optional, auto-detected)

All functions are pure — no dependency on IOEnvironment state.
"""

from __future__ import annotations
import os, re, shutil, importlib, traceback
from dataclasses import dataclass, field


@dataclass
class ProfileResult:
    """Runtime profile metrics for a single kernel."""
    kernel_time_ms: float = 0.0
    achieved_bandwidth_gbps: float = 0.0
    peak_bandwidth_gbps: float = 0.0
    bandwidth_utilization_pct: float = 0.0
    achieved_tflops: float = 0.0
    peak_tflops: float = 0.0
    compute_utilization_pct: float = 0.0
    uses_tensor_core: bool = False
    tensor_core_hint: str = ""
    roofline_bottleneck: str = ""  # "memory-bound", "compute-bound", "under-utilized"
    kernel_names: list[str] = field(default_factory=list)
    raw_details: str = ""

    def display(self, label: str = "Kernel") -> str:
        lines = [
            f"┌─── Runtime Profile: {label} ────────────────────┐",
            f"│  Kernel time:       {self.kernel_time_ms:>8.3f} ms                │",
            f"│  Bandwidth:         {self.achieved_bandwidth_gbps:>8.1f} / {self.peak_bandwidth_gbps:.0f} GB/s ({self.bandwidth_utilization_pct:.0f}%)   │",
            f"│  Compute:           {self.achieved_tflops:>8.2f} / {self.peak_tflops:.1f} TFLOPS ({self.compute_utilization_pct:.0f}%) │",
            f"│  Tensor Core:       {'YES ✓' if self.uses_tensor_core else 'NO ✗':>8s}                       │",
            f"│  Roofline:          {self.roofline_bottleneck:>20s}       │",
            f"└──────────────────────────────────────────────────┘",
        ]
        return "\n".join(lines)


@dataclass
class CompareResult:
    """Side-by-side comparison of two kernel profiles."""
    baseline: ProfileResult
    flash: ProfileResult
    library: ProfileResult | None = None
    diagnosis: list[str] = field(default_factory=list)

    def display(self) -> str:
        b, f = self.baseline, self.flash
        lines = [
            f"┌─── Compare Profile ──────────────────────────────────────────┐",
            f"│                      {'Baseline':>12s}  {'Flash':>12s}  {'Delta':>8s}     │",
            f"├──────────────────────────────────────────────────────────────┤",
            f"│  Kernel time (ms)   {b.kernel_time_ms:>12.3f}  {f.kernel_time_ms:>12.3f}  {f.kernel_time_ms/max(b.kernel_time_ms,1e-9):>7.2f}×    │",
            f"│  Bandwidth (GB/s)   {b.achieved_bandwidth_gbps:>12.1f}  {f.achieved_bandwidth_gbps:>12.1f}  {f.achieved_bandwidth_gbps/max(b.achieved_bandwidth_gbps,1e-9):>7.2f}×    │",
            f"│  BW util (%)        {b.bandwidth_utilization_pct:>12.0f}  {f.bandwidth_utilization_pct:>12.0f}  {f.bandwidth_utilization_pct - b.bandwidth_utilization_pct:>+7.0f}%    │",
            f"│  Compute (TFLOPS)   {b.achieved_tflops:>12.2f}  {f.achieved_tflops:>12.2f}  {f.achieved_tflops/max(b.achieved_tflops,1e-9):>7.2f}×    │",
            f"│  Compute util (%)   {b.compute_utilization_pct:>12.0f}  {f.compute_utilization_pct:>12.0f}  {f.compute_utilization_pct - b.compute_utilization_pct:>+7.0f}%    │",
            f"│  Tensor Core        {'YES' if b.uses_tensor_core else 'NO':>12s}  {'YES' if f.uses_tensor_core else 'NO':>12s}           │",
            f"│  Bottleneck         {b.roofline_bottleneck:>12s}  {f.roofline_bottleneck:>12s}           │",
        ]
        if self.library:
            lb = self.library
            lines.append(f"├──────────────────────────────────────────────────────────────┤")
            lines.append(f"│  Library            {lb.kernel_time_ms:>12.3f} ms  ({lb.roofline_bottleneck})          │")
            gap = lb.kernel_time_ms / max(f.kernel_time_ms, 1e-9)
            lines.append(f"│  Flash vs Library   {gap:>12.2f}×                            │")

        if self.diagnosis:
            lines.append(f"├──────────────────────────────────────────────────────────────┤")
            for d in self.diagnosis:
                lines.append(f"│  {d:<60s}│")
        lines.append(f"└──────────────────────────────────────────────────────────────┘")
        return "\n".join(lines)


# ============================================================================
# Hardware detection
# ============================================================================

def _detect_hardware():
    """Detect GPU hardware and return peak specs."""
    import torch
    name = torch.cuda.get_device_name()
    if "A100" in name:
        return 2039.0, 19.5, 40.0, 108  # BW GB/s, TFLOPS FP32, L2 MB, SMs
    elif "H100" in name:
        return 3350.0, 67.0, 50.0, 132
    elif "H200" in name:
        return 4800.0, 67.0, 50.0, 132
    elif "4090" in name:
        return 1008.0, 82.6, 72.0, 128
    else:
        # Fallback: read from torch
        props = torch.cuda.get_device_properties(0)
        bw = props.total_memory / 1e9 * 10  # rough estimate
        return bw, 19.5, 40.0, props.multi_processor_count


def _find_ncu() -> str | None:
    """Find ncu binary — check PATH then common CUDA install locations."""
    ncu = shutil.which("ncu")
    if ncu:
        return ncu
    for prefix in ["/usr/local/cuda/bin", "/usr/local/cuda-13.1/bin",
                   "/usr/local/cuda-13/bin", "/usr/local/cuda-12/bin",
                   "/opt/nvidia/nsight-compute/2025.4.1"]:
        candidate = os.path.join(prefix, "ncu")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _has_ncu() -> bool:
    return _find_ncu() is not None


# ============================================================================
# Data volume estimation per task
# ============================================================================

def _estimate_data_bytes(task: str, params: dict) -> tuple[int, int]:
    """Estimate (read_bytes, write_bytes) for the flash kernel."""
    if task == "cross_entropy":
        N, V = params.get("N", 4096), params.get("V", 32000)
        read_bytes = N * V * 4 + N * 4  # logits + labels
        write_bytes = N * 4  # loss
        return read_bytes, write_bytes
    elif task == "kmeans":
        N, K, d = params.get("N", 65536), params.get("K", 1024), params.get("d", 128)
        read_bytes = N * d * 4 + K * d * 4  # X + C
        write_bytes = N * 4  # assignments
        return read_bytes, write_bytes
    elif task == "softmax":
        N = params.get("N", 4096)
        M = params.get("V", params.get("d", 4096))
        read_bytes = N * M * 4 * 2  # 2 passes
        write_bytes = N * M * 4
        return read_bytes, write_bytes
    elif task == "layernorm":
        N, d = params.get("N", 16384), params.get("d", 4096)
        read_bytes = N * d * 4 * 2 + d * 4 * 2  # X (2 passes) + gamma + beta
        write_bytes = N * d * 4
        return read_bytes, write_bytes
    elif task in ("gmm_estep", "gmm_em_fused"):
        N, K, d = params.get("N", 65536), params.get("K", 1024), params.get("d", 128)
        read_bytes = N * d * 4 + K * d * 4 * 2 + K * 4  # X + mu + var + log_pi
        write_bytes = N * 4  # log_norm
        return read_bytes, write_bytes
    elif task == "fft":
        N = params.get("N", 1048576)
        log2N = params.get("log2N", 20)
        log2B = 10  # TILE=1024
        global_passes = max(log2N - log2B, 0)
        read_bytes = N * 4 * 2 * (1 + global_passes)  # re+im, initial + global passes
        write_bytes = N * 4 * 2 * (1 + global_passes)
        return read_bytes, write_bytes
    elif task == "conv2d":
        N = params.get("N", 16)
        C_in, C_out = params.get("C_in", 64), params.get("C_out", 128)
        H, W = params.get("H", 28), params.get("W", 28)
        KH, KW = params.get("KH", 3), params.get("KW", 3)
        OH, OW = params.get("OH", H), params.get("OW", W)
        read_bytes = N * C_in * H * W * 4 + C_out * C_in * KH * KW * 4
        write_bytes = N * C_out * OH * OW * 4
        return read_bytes, write_bytes
    elif task == "stencil2d":
        H, W = params.get("H", 4096), params.get("W", 4096)
        T = params.get("T", 100)
        S = params.get("S", 4)
        passes = (T + S - 1) // S
        read_bytes = passes * H * W * 4
        write_bytes = passes * H * W * 4
        return read_bytes, write_bytes
    else:
        return 0, 0


def _estimate_flops(task: str, params: dict) -> int:
    """Estimate total FLOPs for the flash kernel."""
    if task == "cross_entropy":
        N, V = params.get("N", 4096), params.get("V", 32000)
        return N * V * 5  # exp + compare + add + mul + log per element
    elif task == "kmeans":
        N, K, d = params.get("N", 65536), params.get("K", 1024), params.get("d", 128)
        return N * K * (2 * d + 2)
    elif task == "softmax":
        N = params.get("N", 4096)
        M = params.get("V", params.get("d", 4096))
        return N * M * 5
    elif task == "layernorm":
        N, d = params.get("N", 16384), params.get("d", 4096)
        return N * d * 9
    elif task in ("gmm_estep", "gmm_em_fused"):
        N, K, d = params.get("N", 65536), params.get("K", 1024), params.get("d", 128)
        return N * K * (3 * d + 5)
    elif task == "fft":
        N = params.get("N", 1048576)
        log2N = params.get("log2N", 20)
        return N * log2N * 10
    elif task == "conv2d":
        N = params.get("N", 16)
        C_in, C_out = params.get("C_in", 64), params.get("C_out", 128)
        OH, OW = params.get("OH", 28), params.get("OW", 28)
        KH, KW = params.get("KH", 3), params.get("KW", 3)
        return 2 * N * OH * OW * C_out * C_in * KH * KW
    elif task == "stencil2d":
        H, W = params.get("H", 4096), params.get("W", 4096)
        T = params.get("T", 100)
        return T * H * W * 5
    else:
        return 0


# ============================================================================
# Core profiling: torch.profiler based
# ============================================================================

def runtime_profile(task: str, params: dict, kernel_path: str,
                    n_warmup: int = 5, n_iter: int = 20) -> ProfileResult:
    """
    Profile a kernel using torch.profiler + timing.

    Returns ProfileResult with bandwidth/compute utilization metrics.
    """
    import torch

    peak_bw, peak_tflops, _, _ = _detect_hardware()

    # Load the kernel module
    spec = importlib.util.spec_from_file_location(f"flash_{task}", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    flash_fn = None
    for name in dir(mod):
        if name.startswith("flash_"):
            flash_fn = getattr(mod, name)
            break
    if not flash_fn:
        return ProfileResult(raw_details="ERROR: No flash_* function found")

    # Construct inputs (reuse triton_codegen patterns)
    inputs = _make_inputs(task, params)
    if inputs is None:
        return ProfileResult(raw_details=f"ERROR: No input constructor for '{task}'")

    # Warmup
    for _ in range(n_warmup):
        flash_fn(*inputs)
    torch.cuda.synchronize()

    # Precise timing
    times = []
    for _ in range(n_iter):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        flash_fn(*inputs)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    kernel_time_ms = sum(times) / len(times)

    # Estimate utilization
    read_bytes, write_bytes = _estimate_data_bytes(task, params)
    total_bytes = read_bytes + write_bytes
    total_flops = _estimate_flops(task, params)

    achieved_bw = total_bytes / (kernel_time_ms / 1000) / 1e9 if kernel_time_ms > 0 else 0
    bw_util = achieved_bw / peak_bw * 100 if peak_bw > 0 else 0

    achieved_tflops = total_flops / (kernel_time_ms / 1000) / 1e12 if kernel_time_ms > 0 else 0
    compute_util = achieved_tflops / peak_tflops * 100 if peak_tflops > 0 else 0

    # Detect Tensor Core usage via torch.profiler kernel names
    uses_tc = False
    tc_hint = ""
    kernel_names = []
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
        ) as prof:
            flash_fn(*inputs)
            torch.cuda.synchronize()
        for evt in prof.key_averages():
            if evt.device_type == torch.autograd.DeviceType.CUDA:
                kname = evt.key
                kernel_names.append(kname)
                if any(tc in kname.lower() for tc in ("hmma", "mma", "tensor", "16816", "wmma")):
                    uses_tc = True
                    tc_hint = f"Tensor Core kernel detected: {kname[:60]}"
    except Exception:
        pass  # torch.profiler may not be available in all envs

    if not uses_tc:
        # Also check the source code
        if os.path.exists(kernel_path):
            with open(kernel_path) as f:
                code = f.read()
            if "tl.dot" in code:
                tc_hint = "tl.dot found in code — Tensor Core expected at runtime"
                uses_tc = True  # Triton tl.dot maps to TC
            else:
                tc_hint = "No tl.dot / Tensor Core usage detected"

    # Roofline classification
    if bw_util > 60 and compute_util < 30:
        bottleneck = "memory-bound"
    elif compute_util > 40 and bw_util < 30:
        bottleneck = "compute-bound"
    elif bw_util < 25 and compute_util < 15:
        bottleneck = "under-utilized"
    else:
        bottleneck = f"balanced ({bw_util:.0f}%BW, {compute_util:.0f}%C)"

    result = ProfileResult(
        kernel_time_ms=kernel_time_ms,
        achieved_bandwidth_gbps=achieved_bw,
        peak_bandwidth_gbps=peak_bw,
        bandwidth_utilization_pct=min(bw_util, 100),
        achieved_tflops=achieved_tflops,
        peak_tflops=peak_tflops,
        compute_utilization_pct=min(compute_util, 100),
        uses_tensor_core=uses_tc,
        tensor_core_hint=tc_hint,
        roofline_bottleneck=bottleneck,
        kernel_names=kernel_names[:5],
    )

    torch.cuda.empty_cache()
    return result


# ============================================================================
# NCU-based profiling (hardware counters, more accurate)
# ============================================================================

def ncu_profile(task: str, params: dict, kernel_path: str) -> str | None:
    """
    Run ncu on a small benchmark script to get hardware counter metrics.
    Returns raw ncu output text, or None if ncu is not available.
    """
    import subprocess, tempfile

    ncu_bin = _find_ncu()
    if not ncu_bin:
        return None

    # Build a small Python script that runs the kernel once
    script = f'''
import torch, importlib, sys
spec = importlib.util.spec_from_file_location("kern", "{kernel_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
flash_fn = None
for name in dir(mod):
    if name.startswith("flash_"):
        flash_fn = getattr(mod, name)
        break
torch.manual_seed(42)
'''
    # Add input construction
    if task == "cross_entropy":
        N, V = params.get("N", 4096), params.get("V", 32000)
        script += f'inputs = (torch.randn({N},{V},device="cuda"), torch.randint(0,{V},({N},),device="cuda"))\n'
    elif task == "kmeans":
        N, K, d = params.get("N", 65536), params.get("K", 1024), params.get("d", 128)
        script += f'inputs = (torch.randn({N},{d},device="cuda"), torch.randn({K},{d},device="cuda"))\n'
    elif task == "softmax":
        N = params.get("N", 4096)
        M = params.get("V", params.get("d", 4096))
        script += f'inputs = (torch.randn({N},{M},device="cuda"),)\n'
    elif task == "layernorm":
        N, d = params.get("N", 16384), params.get("d", 4096)
        script += f'inputs = (torch.randn({N},{d},device="cuda"), torch.ones({d},device="cuda"), torch.zeros({d},device="cuda"))\n'
    elif task == "fft":
        N = params.get("N", min(params.get("N", 65536), 65536))
        script += f'inputs = (torch.randn({N},device="cuda"), torch.zeros({N},device="cuda"))\n'
    elif task == "conv2d":
        N_b = min(params.get("N", 4), 4)
        C_in, C_out = params.get("C_in", 32), params.get("C_out", 64)
        H, W = min(params.get("H", 28), 28), min(params.get("W", 28), 28)
        KH, KW = params.get("KH", 3), params.get("KW", 3)
        script += f'inputs = (torch.randn({N_b},{C_in},{H},{W},device="cuda"), torch.randn({C_out},{C_in},{KH},{KW},device="cuda"))\n'
    elif task == "stencil2d":
        H, W = min(params.get("H", 512), 512), min(params.get("W", 512), 512)
        T = min(params.get("T", 4), 4)
        script += f'inputs = (torch.randn({H},{W},device="cuda"), {T})\n'
    else:
        return None

    script += '''
# warmup
for _ in range(3):
    flash_fn(*inputs)
torch.cuda.synchronize()
# profiled run
flash_fn(*inputs)
torch.cuda.synchronize()
'''

    # Write script to temp file
    script_path = os.path.join(tempfile.gettempdir(), f"ncu_bench_{task}.py")
    with open(script_path, "w") as f:
        f.write(script)

    # Run ncu with key metrics
    metrics = [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "dram__bytes.sum",
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
        "gpu__time_duration.sum",
    ]

    import sys
    python_bin = sys.executable

    # Always try sudo first (GPU perf counters usually need it)
    sudo = shutil.which("sudo")
    base_cmd = [
        ncu_bin, "--target-processes", "all",
        "--metrics", ",".join(metrics),
        "--kernel-name", "regex:.*",
        "--launch-skip", "3",  # skip warmup
        "--launch-count", "1",
        python_bin, script_path,
    ]
    cmd = [sudo] + base_cmd if sudo else base_cmd

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            env={**os.environ, "PATH": f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"},
        )
        output = result.stdout + result.stderr

        # If sudo ncu still fails (no passwordless sudo), try without sudo
        if "ERR_NVGPUCTRPERM" in output and sudo:
            result2 = subprocess.run(
                base_cmd, capture_output=True, text=True, timeout=120,
                env={**os.environ, "PATH": f"/usr/local/cuda/bin:{os.environ.get('PATH', '')}"},
            )
            output2 = result2.stdout + result2.stderr
            if "ERR_NVGPUCTRPERM" not in output2 and output2.strip():
                output = output2

        # Clean up
        try:
            os.unlink(script_path)
        except OSError:
            pass

        if "ERR_NVGPUCTRPERM" in output:
            return ("ncu permission denied: GPU performance counters require elevated access.\n"
                    "Fix: sudo sh -c 'echo 0 > /proc/driver/nvidia/params/RmProfilingAdminOnly'\n"
                    "Or:  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | "
                    "sudo tee /etc/modprobe.d/nvidia-perf.conf && reboot\n"
                    "Falling back to torch.profiler estimates.")

        return output if output.strip() else None
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError) as e:
        return f"ncu error: {e}"


# ============================================================================
# Library ceiling benchmark
# ============================================================================

def library_ceiling(task: str, params: dict,
                    n_warmup: int = 5, n_iter: int = 20) -> dict:
    """
    Benchmark the library-optimized implementation (cuDNN/cuFFT/cuBLAS/ATen)
    as a performance ceiling.

    Returns dict with library_time_ms, library_name, and comparison info.
    """
    import torch
    import torch.nn.functional as F

    inputs = _make_inputs(task, params)
    if inputs is None:
        return {"error": f"No input constructor for '{task}'"}

    # Select library function
    lib_fn = None
    lib_name = "PyTorch/ATen"

    if task == "cross_entropy":
        lib_name = "PyTorch F.cross_entropy"
        lib_fn = lambda: F.cross_entropy(inputs[0], inputs[1], reduction="none")
    elif task == "kmeans":
        lib_name = "cuBLAS (torch.cdist+argmin)"
        lib_fn = lambda: torch.cdist(inputs[0], inputs[1], p=2.0).argmin(dim=1)
    elif task == "softmax":
        lib_name = "ATen torch.softmax"
        lib_fn = lambda: torch.softmax(inputs[0], dim=1)
    elif task == "layernorm":
        lib_name = "ATen F.layer_norm"
        lib_fn = lambda: F.layer_norm(inputs[0], [inputs[0].shape[1]], inputs[1], inputs[2])
    elif task in ("gmm_estep", "gmm_em_fused"):
        lib_name = "ATen (broadcast matmul)"
        X, mu, var, log_pi = inputs
        d = X.shape[1]
        def _lib_gmm():
            log_det = var.log().sum(1)
            diff = X.unsqueeze(1) - mu.unsqueeze(0)
            mahal = (diff ** 2 / var.unsqueeze(0)).sum(2)
            L = log_pi.unsqueeze(0) - 0.5 * (d * 1.8379 + log_det.unsqueeze(0) + mahal)
            return torch.logsumexp(L, 1)
        lib_fn = _lib_gmm
    elif task == "fft":
        lib_name = "cuFFT (torch.fft.fft)"
        lib_fn = lambda: torch.fft.fft(torch.complex(inputs[0], inputs[1]))
    elif task == "conv2d":
        lib_name = "cuDNN (torch.conv2d)"
        inp, wt = inputs
        KH = wt.shape[2]
        pad = KH // 2
        lib_fn = lambda: F.conv2d(inp, wt, padding=pad)
    elif task == "stencil2d":
        # No standard library for stencil
        return {"library_name": "none", "library_time_ms": float("inf"),
                "note": "No standard library for stencil operations"}
    else:
        return {"error": f"No library function for '{task}'"}

    # Warmup
    for _ in range(n_warmup):
        lib_fn()
    torch.cuda.synchronize()

    # Benchmark
    s_ev = torch.cuda.Event(enable_timing=True)
    e_ev = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n_iter):
        s_ev.record()
        lib_fn()
        e_ev.record()
        torch.cuda.synchronize()
        times.append(s_ev.elapsed_time(e_ev))

    lib_time = sum(times) / len(times)
    torch.cuda.empty_cache()

    return {
        "library_name": lib_name,
        "library_time_ms": lib_time,
    }


# ============================================================================
# Compare profile: baseline vs flash (vs library)
# ============================================================================

def compare_profile(task: str, params: dict, kernel_path: str,
                    n_warmup: int = 5, n_iter: int = 20) -> CompareResult:
    """
    Profile baseline (materialized PyTorch) and flash kernel side-by-side.
    Optionally includes library ceiling.
    """
    import torch

    peak_bw, peak_tflops, _, _ = _detect_hardware()
    inputs = _make_inputs(task, params)
    if inputs is None:
        return CompareResult(
            baseline=ProfileResult(raw_details="ERROR: no inputs"),
            flash=ProfileResult(raw_details="ERROR: no inputs"),
        )

    read_bytes, write_bytes = _estimate_data_bytes(task, params)
    total_bytes = read_bytes + write_bytes
    total_flops = _estimate_flops(task, params)

    def _profile_fn(fn, label):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(n_iter):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        t = sum(times) / len(times)
        bw = total_bytes / (t / 1000) / 1e9 if t > 0 else 0
        tflops = total_flops / (t / 1000) / 1e12 if t > 0 else 0
        bw_pct = min(bw / peak_bw * 100, 100) if peak_bw > 0 else 0
        c_pct = min(tflops / peak_tflops * 100, 100) if peak_tflops > 0 else 0

        if bw_pct > 60 and c_pct < 30:
            bn = "memory-bound"
        elif c_pct > 40 and bw_pct < 30:
            bn = "compute-bound"
        elif bw_pct < 25 and c_pct < 15:
            bn = "under-utilized"
        else:
            bn = "balanced"

        return ProfileResult(
            kernel_time_ms=t,
            achieved_bandwidth_gbps=bw, peak_bandwidth_gbps=peak_bw,
            bandwidth_utilization_pct=bw_pct,
            achieved_tflops=tflops, peak_tflops=peak_tflops,
            compute_utilization_pct=c_pct,
            roofline_bottleneck=bn,
        )

    # Baseline
    baseline_fn = _make_baseline_fn(task, inputs)
    baseline_prof = _profile_fn(baseline_fn, "Baseline")

    # Flash
    spec = importlib.util.spec_from_file_location(f"flash_{task}", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    flash_fn = None
    for name in dir(mod):
        if name.startswith("flash_"):
            flash_fn = getattr(mod, name)
            break
    flash_prof = _profile_fn(lambda: flash_fn(*inputs), "Flash")

    # Library
    lib_prof = None
    lib_info = library_ceiling(task, params, n_warmup, n_iter)
    if "library_time_ms" in lib_info and lib_info["library_time_ms"] < float("inf"):
        lib_prof = ProfileResult(
            kernel_time_ms=lib_info["library_time_ms"],
            peak_bandwidth_gbps=peak_bw, peak_tflops=peak_tflops,
            roofline_bottleneck=lib_info["library_name"],
        )

    # Diagnosis
    diagnosis = []
    if baseline_prof.uses_tensor_core and not flash_prof.uses_tensor_core:
        diagnosis.append("⚠ CRITICAL: Baseline uses Tensor Core but Flash does NOT")
    if flash_prof.bandwidth_utilization_pct < 25 and flash_prof.compute_utilization_pct < 15:
        diagnosis.append("⚠ Flash kernel is under-utilized — structural parallelism problem")
    if flash_prof.bandwidth_utilization_pct > 70:
        diagnosis.append("✓ Flash kernel is memory-bound with good BW utilization")
    if lib_prof and lib_prof.kernel_time_ms > 0:
        gap = lib_prof.kernel_time_ms / max(flash_prof.kernel_time_ms, 1e-9)
        if gap > 5:
            diagnosis.append(f"⚠ Library is {1/gap:.0f}× faster — gap too large for hand-written kernel")
        elif gap > 0.9:
            diagnosis.append(f"✓ Flash is within {1/gap:.1f}× of library — near optimal")

    torch.cuda.empty_cache()
    return CompareResult(baseline=baseline_prof, flash=flash_prof,
                         library=lib_prof, diagnosis=diagnosis)


# ============================================================================
# Occupancy estimation
# ============================================================================

def estimate_occupancy(code: str) -> dict:
    """
    Estimate kernel occupancy from source code.
    Static analysis of BLOCK_SIZE, num_warps, shared memory usage.
    """
    import re

    # Extract num_warps
    m = re.search(r'num_warps\s*=\s*(\d+)', code)
    num_warps = int(m.group(1)) if m else 4

    # Extract BLOCK_SIZE / BLOCK_M
    m = re.search(r'BLOCK_(?:SIZE|M)\s*[=:]\s*(\d+)', code)
    block_size = int(m.group(1)) if m else 128

    # Threads per block
    threads_per_block = num_warps * 32

    # Estimate registers (Triton typically uses 40-128 per thread)
    has_tl_dot = "tl.dot" in code
    has_complex_online = code.count("running_") >= 2 or code.count("best_") >= 2
    if has_tl_dot:
        est_regs = 96  # tl.dot kernels tend to use more registers
    elif has_complex_online:
        est_regs = 72
    else:
        est_regs = 48

    # Estimate shared memory
    # Look for explicit sizes or infer from block sizes
    m_smem = re.search(r'shared.*?(\d+)\s*\*\s*sizeof\(float\)', code, re.IGNORECASE)
    if m_smem:
        smem_bytes = int(m_smem.group(1)) * 4
    else:
        # Triton auto-allocates SMEM for tl.dot operands
        if has_tl_dot:
            bk = 64  # typical
            m_bk = re.search(r'BK\s*[=:]\s*(\d+)', code)
            if m_bk:
                bk = int(m_bk.group(1))
            smem_bytes = 2 * block_size * bk * 4  # two operand tiles
        else:
            smem_bytes = block_size * 4 * 4  # minimal

    # A100 limits
    max_regs_per_sm = 65536
    max_smem_per_sm = 164 * 1024  # A100
    max_threads_per_sm = 2048
    max_blocks_per_sm = 32

    # Compute limits
    blocks_by_regs = max_regs_per_sm // (est_regs * threads_per_block) if est_regs * threads_per_block > 0 else 32
    blocks_by_smem = max_smem_per_sm // smem_bytes if smem_bytes > 0 else 32
    blocks_by_threads = max_threads_per_sm // threads_per_block if threads_per_block > 0 else 32
    active_blocks = min(blocks_by_regs, blocks_by_smem, blocks_by_threads, max_blocks_per_sm)

    active_warps = active_blocks * num_warps
    max_warps = max_threads_per_sm // 32
    occupancy_pct = active_warps / max_warps * 100 if max_warps > 0 else 0

    limiting = "registers" if blocks_by_regs <= blocks_by_smem and blocks_by_regs <= blocks_by_threads else \
               "shared_memory" if blocks_by_smem <= blocks_by_threads else "threads"

    return {
        "block_size": block_size,
        "num_warps": num_warps,
        "threads_per_block": threads_per_block,
        "est_registers_per_thread": est_regs,
        "est_smem_bytes": smem_bytes,
        "active_blocks_per_sm": active_blocks,
        "active_warps_per_sm": active_warps,
        "max_warps_per_sm": max_warps,
        "occupancy_pct": occupancy_pct,
        "limiting_factor": limiting,
    }


# ============================================================================
# Input constructors (shared between profile functions)
# ============================================================================

def _make_inputs(task: str, params: dict):
    """Construct GPU tensors for the given task. Returns tuple of tensors or None."""
    import torch
    torch.manual_seed(42)

    if task == "cross_entropy":
        N, V = params.get("N", 4096), params.get("V", 32000)
        return (torch.randn(N, V, device="cuda"), torch.randint(0, V, (N,), device="cuda"))
    elif task == "kmeans":
        N, K, d = params.get("N", 65536), params.get("K", 1024), params.get("d", 128)
        return (torch.randn(N, d, device="cuda"), torch.randn(K, d, device="cuda"))
    elif task == "softmax":
        N = params.get("N", 4096)
        M = params.get("V", params.get("d", 4096))
        return (torch.randn(N, M, device="cuda"),)
    elif task == "layernorm":
        N, d = params.get("N", 16384), params.get("d", 4096)
        return (torch.randn(N, d, device="cuda"), torch.ones(d, device="cuda"), torch.zeros(d, device="cuda"))
    elif task in ("gmm_estep", "gmm_em_fused"):
        N, K, d = params.get("N", 16384), params.get("K", 64), params.get("d", 64)
        return (torch.randn(N, d, device="cuda"), torch.randn(K, d, device="cuda"),
                torch.ones(K, d, device="cuda"),
                torch.full((K,), -torch.log(torch.tensor(float(K))), device="cuda"))
    elif task == "fft":
        N = params.get("N", 1048576)
        return (torch.randn(N, device="cuda"), torch.zeros(N, device="cuda"))
    elif task == "conv2d":
        N = params.get("N", 16)
        C_in, C_out = params.get("C_in", 64), params.get("C_out", 128)
        H, W = params.get("H", 28), params.get("W", 28)
        KH, KW = params.get("KH", 3), params.get("KW", 3)
        return (torch.randn(N, C_in, H, W, device="cuda"),
                torch.randn(C_out, C_in, KH, KW, device="cuda"))
    elif task == "stencil2d":
        H, W = params.get("H", 4096), params.get("W", 4096)
        T = params.get("T", 100)
        return (torch.randn(H, W, device="cuda"), T)
    return None


def _make_baseline_fn(task: str, inputs):
    """Return a callable baseline (materialized) function."""
    import torch
    import torch.nn.functional as F

    if task == "cross_entropy":
        logits, labels = inputs
        def fn():
            m = logits.max(dim=1, keepdim=True).values
            exp = (logits - m).exp()
            lse = exp.sum(dim=1).log()
            g = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            return -(g - m.squeeze(1) - lse)
        return fn
    elif task == "kmeans":
        X, C = inputs
        return lambda: torch.cdist(X, C, p=2.0).argmin(dim=1)
    elif task == "softmax":
        return lambda: torch.softmax(inputs[0], dim=1)
    elif task == "layernorm":
        X, g, b = inputs
        def fn():
            m = X.mean(dim=1, keepdim=True)
            v = X.var(dim=1, keepdim=True, unbiased=False)
            return (X - m) / torch.sqrt(v + 1e-5) * g + b
        return fn
    elif task in ("gmm_estep", "gmm_em_fused"):
        X, mu, var, log_pi = inputs
        d = X.shape[1]
        def fn():
            log_det = var.log().sum(1)
            diff = X.unsqueeze(1) - mu.unsqueeze(0)
            mahal = (diff ** 2 / var.unsqueeze(0)).sum(2)
            L = log_pi.unsqueeze(0) - 0.5 * (d * 1.8379 + log_det.unsqueeze(0) + mahal)
            return torch.logsumexp(L, 1)
        return fn
    elif task == "fft":
        x_re, x_im = inputs
        return lambda: torch.fft.fft(torch.complex(x_re, x_im))
    elif task == "conv2d":
        inp, wt = inputs
        KH = wt.shape[2]
        pad = KH // 2
        def fn():
            col = F.unfold(inp, (KH, KH), padding=(pad, pad))
            wt_flat = wt.reshape(wt.shape[0], -1)
            return wt_flat @ col
        return fn
    elif task == "stencil2d":
        grid, T = inputs
        def fn():
            g = grid.clone()
            out = torch.empty_like(g)
            for _ in range(T):
                out[1:-1, 1:-1] = 0.2 * (g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
                g, out = out, g
            return g
        return fn
    return lambda: None
