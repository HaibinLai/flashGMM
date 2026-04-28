"""
Computation Graph DSL for IO complexity analysis.

Provides structured descriptions of operator data flow,
marking each tensor's storage location (HBM vs on-chip).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import copy


@dataclass
class TensorSpec:
    """Specification of a tensor in the computation graph."""
    shape: tuple          # symbolic shape, e.g. ("N", "K") or (1024, 128)
    dtype: str = "f32"    # f32, f16, bf16
    storage: str = "HBM"  # "HBM" | "on_chip" | "register"

    @property
    def dtype_bytes(self) -> int:
        return {"f32": 4, "f16": 2, "bf16": 2, "f64": 8, "i32": 4}[self.dtype]

    def num_elements(self, params: dict) -> int:
        """Resolve symbolic shape to concrete element count."""
        total = 1
        for s in self.shape:
            if isinstance(s, str):
                total *= params[s]
            else:
                total *= s
        return total

    def size_bytes(self, params: dict) -> int:
        return self.num_elements(params) * self.dtype_bytes

    def __repr__(self):
        shape_str = ", ".join(str(s) for s in self.shape)
        return f"Tensor([{shape_str}], {self.dtype}, {self.storage})"


@dataclass
class TilingSpec:
    """Tiling configuration for a fused operation."""
    tiles: dict  # e.g. {"N": "BN", "K": "BK"}

    def tile_count(self, dim: str, params: dict) -> int:
        tile_size = params[self.tiles[dim]]
        full_size = params[dim]
        return (full_size + tile_size - 1) // tile_size


@dataclass
class OnlineStateSpec:
    """Online algorithm state maintained in registers."""
    variables: dict       # name -> type, e.g. {"running_max": "scalar", "running_sum_exp": "scalar"}
    algorithm: str        # "online_logsumexp" | "online_argmin" | "online_softmax"
    output_shape: tuple   # shape of the final output, e.g. ("N",)

    def state_size_per_point(self) -> int:
        """Bytes of register state per data point."""
        sizes = {"scalar": 4, "vector_d": None}  # vector_d resolved later
        return sum(sizes[v] for v in self.variables.values() if sizes[v] is not None)


@dataclass
class Op:
    """A single operation in the computation graph."""
    name: str
    reads: list[str]              # names of input tensors to read from HBM
    computes: str                 # description of computation type
    flops_fn: Callable            # function(params) -> int, FLOPs count
    output: TensorSpec            # output tensor spec
    output_name: str              # name to reference this output
    is_fused: bool = False
    tiling: TilingSpec | None = None
    online_state: OnlineStateSpec | None = None
    notes: str = ""               # human-readable note

    def flops(self, params: dict) -> int:
        return self.flops_fn(params)


@dataclass
class FusedOp(Op):
    """A fused operation that processes data in tiles without materializing intermediates."""
    def __init__(self, name, reads, computes, flops_fn, output, output_name,
                 tiling=None, online_state=None, notes=""):
        super().__init__(
            name=name, reads=reads, computes=computes, flops_fn=flops_fn,
            output=output, output_name=output_name, is_fused=True,
            tiling=tiling, online_state=online_state, notes=notes
        )


@dataclass
class ComputationGraph:
    """
    A directed acyclic graph of operations describing an operator's data flow.
    
    Each operation reads tensors (from HBM or previous ops), computes something,
    and produces an output tensor (stored in HBM or kept on-chip).
    """
    name: str
    inputs: dict[str, TensorSpec]     # name -> TensorSpec
    operations: list[Op]
    outputs: list[str]                # names of final output tensors
    description: str = ""

    def get_tensor(self, name: str) -> TensorSpec:
        """Look up a tensor by name (input or intermediate)."""
        if name in self.inputs:
            return self.inputs[name]
        for op in self.operations:
            if op.output_name == name:
                return op.output
        raise KeyError(f"Tensor '{name}' not found in graph")

    def get_all_tensors(self) -> dict[str, TensorSpec]:
        """Return all tensors (inputs + intermediates + outputs)."""
        tensors = dict(self.inputs)
        for op in self.operations:
            tensors[op.output_name] = op.output
        return tensors

    def get_materialized_intermediates(self) -> list[tuple[str, TensorSpec]]:
        """Find intermediate tensors stored in HBM (not final outputs)."""
        intermediates = []
        for op in self.operations:
            if op.output_name not in self.outputs and op.output.storage == "HBM":
                intermediates.append((op.output_name, op.output))
        return intermediates

    def clone(self) -> ComputationGraph:
        return copy.deepcopy(self)

    def summary(self, params: dict) -> str:
        """Human-readable summary of the computation graph."""
        lines = [f"=== {self.name} ==="]
        if self.description:
            lines.append(f"  {self.description}")
        lines.append(f"  Inputs:")
        for name, spec in self.inputs.items():
            lines.append(f"    {name}: {spec}  ({spec.size_bytes(params)/1e6:.1f} MB)")
        lines.append(f"  Operations:")
        for op in self.operations:
            storage_tag = f"[{op.output.storage}]"
            is_intermediate = op.output_name not in self.outputs
            mat_tag = " *** MATERIALIZED ***" if (is_intermediate and op.output.storage == "HBM") else ""
            fused_tag = " [FUSED]" if op.is_fused else ""
            lines.append(
                f"    {op.name}: reads {op.reads} → {op.output_name} "
                f"{op.output} ({op.output.size_bytes(params)/1e6:.1f} MB) "
                f"{storage_tag}{fused_tag}{mat_tag}"
            )
            if op.online_state:
                lines.append(f"      online: {op.online_state.algorithm}, state={op.online_state.variables}")
            if op.notes:
                lines.append(f"      note: {op.notes}")
        lines.append(f"  Outputs: {self.outputs}")

        mats = self.get_materialized_intermediates()
        if mats:
            total_mat = sum(s.size_bytes(params) for _, s in mats)
            lines.append(f"  ⚠ Materialized intermediates: {[n for n,_ in mats]} ({total_mat/1e6:.1f} MB)")
        else:
            lines.append(f"  ✓ No materialized intermediates")
        return "\n".join(lines)
