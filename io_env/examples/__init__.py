"""
Pre-built computation graph examples for testing the IO environment.

Includes baseline and optimized versions of:
  - GMM E-step
  - KMeans Assignment
  - Softmax
  - Cross-Entropy Loss
"""

from ..dsl import ComputationGraph, TensorSpec, Op, FusedOp, TilingSpec, OnlineStateSpec


# ============================================================================
# GMM E-step
# ============================================================================

def gmm_estep_baseline() -> ComputationGraph:
    """
    Standard GMM E-step: materializes L ∈ R^{N×K} and γ ∈ R^{N×K}.
    
    IO: 3 × Θ(NK) (write L + read L + write γ) + O(Nd + Kd)
    """
    return ComputationGraph(
        name="gmm_estep_standard",
        description="Standard GMM E-step with full L and γ materialization",
        inputs={
            "X":      TensorSpec(shape=("N", "d"), dtype="f32"),
            "mu":     TensorSpec(shape=("K", "d"), dtype="f32"),
            "var":    TensorSpec(shape=("K", "d"), dtype="f32"),
            "log_pi": TensorSpec(shape=("K",),     dtype="f32"),
        },
        operations=[
            Op(name="compute_L",
               reads=["X", "mu", "var", "log_pi"],
               computes="log_likelihood",
               flops_fn=lambda p: p["N"] * p["K"] * (3 * p["d"] + 2),
               output=TensorSpec(shape=("N", "K"), dtype="f32", storage="HBM"),
               output_name="L",
               notes="Materialize full log-likelihood matrix to HBM"),

            Op(name="logsumexp",
               reads=["L"],
               computes="row_reduce",
               flops_fn=lambda p: p["N"] * p["K"] * 3,
               output=TensorSpec(shape=("N",), dtype="f32", storage="HBM"),
               output_name="log_norm"),

            Op(name="normalize",
               reads=["L", "log_norm"],
               computes="elementwise",
               flops_fn=lambda p: p["N"] * p["K"] * 2,
               output=TensorSpec(shape=("N", "K"), dtype="f32", storage="HBM"),
               output_name="gamma"),
        ],
        outputs=["gamma", "log_norm"],
    )


def gmm_estep_flash() -> ComputationGraph:
    """
    Flash GMM E-step: online log-sum-exp, no L materialization.
    
    IO: O(Nd + Kd + N) — eliminates all N×K traffic.
    """
    return ComputationGraph(
        name="gmm_estep_flash",
        description="Flash GMM E-step: online log-sum-exp, zero N×K materialization",
        inputs={
            "X":      TensorSpec(shape=("N", "d"), dtype="f32"),
            "mu":     TensorSpec(shape=("K", "d"), dtype="f32"),
            "var":    TensorSpec(shape=("K", "d"), dtype="f32"),
            "log_pi": TensorSpec(shape=("K",),     dtype="f32"),
        },
        operations=[
            FusedOp(
                name="flash_log_normalizer",
                reads=["X", "mu", "var", "log_pi"],
                computes="log_likelihood+logsumexp[online_logsumexp]",
                flops_fn=lambda p: p["N"] * p["K"] * (3 * p["d"] + 5),
                output=TensorSpec(shape=("N",), dtype="f32", storage="HBM"),
                output_name="log_norm",
                tiling=TilingSpec(tiles={"N": "BN", "K": "BK"}),
                online_state=OnlineStateSpec(
                    variables={"running_max": "scalar", "running_sum_exp": "scalar"},
                    algorithm="online_logsumexp",
                    output_shape=("N",),
                ),
                notes="Fused compute_L + logsumexp via online log-sum-exp. "
                      "L never materialized. Output is O(N) not O(NK).",
            ),
        ],
        outputs=["log_norm"],
    )


def gmm_em_fused_flash() -> ComputationGraph:
    """
    Flash GMM E+M fused: two passes, zero N×K materialization.
    
    Pass 1: online log-sum-exp → log_normalizer (O(N))
    Pass 2: recompute γ on-chip, accumulate sufficient statistics (O(Kd))
    """
    return ComputationGraph(
        name="gmm_em_fused_flash",
        description="Flash GMM E+M fused: 2-pass, zero N×K materialization",
        inputs={
            "X":      TensorSpec(shape=("N", "d"), dtype="f32"),
            "mu":     TensorSpec(shape=("K", "d"), dtype="f32"),
            "var":    TensorSpec(shape=("K", "d"), dtype="f32"),
            "log_pi": TensorSpec(shape=("K",),     dtype="f32"),
        },
        operations=[
            # Pass 1
            FusedOp(
                name="flash_log_normalizer",
                reads=["X", "mu", "var", "log_pi"],
                computes="log_likelihood+logsumexp[online_logsumexp]",
                flops_fn=lambda p: p["N"] * p["K"] * (3 * p["d"] + 5),
                output=TensorSpec(shape=("N",), dtype="f32", storage="HBM"),
                output_name="log_norm",
                tiling=TilingSpec(tiles={"N": "BN", "K": "BK"}),
                online_state=OnlineStateSpec(
                    variables={"running_max": "scalar", "running_sum_exp": "scalar"},
                    algorithm="online_logsumexp",
                    output_shape=("N",),
                ),
                notes="Pass 1: streaming online log-sum-exp",
            ),
            # Pass 2
            FusedOp(
                name="flash_accumulate_stats",
                reads=["X", "mu", "var", "log_pi", "log_norm"],
                computes="recompute_gamma+accumulate_stats",
                flops_fn=lambda p: p["N"] * p["K"] * (3 * p["d"] + 5 + p["d"] * 3),
                output=TensorSpec(shape=("K", "d"), dtype="f32", storage="HBM"),
                output_name="sufficient_stats",
                tiling=TilingSpec(tiles={"N": "BN", "K": "BK"}),
                notes="Pass 2: recompute γ on-chip + accumulate n_k, s_k, sq_k. "
                      "Writes only O(Kd) sufficient statistics.",
            ),
        ],
        outputs=["log_norm", "sufficient_stats"],
    )


# ============================================================================
# KMeans Assignment
# ============================================================================

def kmeans_assignment_baseline() -> ComputationGraph:
    """Standard KMeans: materialize distance matrix D ∈ R^{N×K}."""
    return ComputationGraph(
        name="kmeans_assign_standard",
        description="Standard KMeans assignment with distance matrix materialization",
        inputs={
            "X": TensorSpec(shape=("N", "d"), dtype="f32"),
            "C": TensorSpec(shape=("K", "d"), dtype="f32"),
        },
        operations=[
            Op(name="compute_distances",
               reads=["X", "C"],
               computes="pairwise_l2",
               flops_fn=lambda p: p["N"] * p["K"] * (2 * p["d"] + 1),
               output=TensorSpec(shape=("N", "K"), dtype="f32", storage="HBM"),
               output_name="D",
               notes="Materialize full N×K distance matrix"),

            Op(name="argmin",
               reads=["D"],
               computes="row_argmin",
               flops_fn=lambda p: p["N"] * p["K"],
               output=TensorSpec(shape=("N",), dtype="i32", storage="HBM"),
               output_name="assignments"),
        ],
        outputs=["assignments"],
    )


def kmeans_assignment_flash() -> ComputationGraph:
    """Flash-KMeans: online argmin, no D materialization."""
    return ComputationGraph(
        name="kmeans_assign_flash",
        description="Flash-KMeans: online argmin, zero distance matrix materialization",
        inputs={
            "X": TensorSpec(shape=("N", "d"), dtype="f32"),
            "C": TensorSpec(shape=("K", "d"), dtype="f32"),
        },
        operations=[
            FusedOp(
                name="flash_assign",
                reads=["X", "C"],
                computes="pairwise_l2+argmin[online_argmin]",
                flops_fn=lambda p: p["N"] * p["K"] * (2 * p["d"] + 2),
                output=TensorSpec(shape=("N",), dtype="i32", storage="HBM"),
                output_name="assignments",
                tiling=TilingSpec(tiles={"N": "BN", "K": "BK"}),
                online_state=OnlineStateSpec(
                    variables={"running_min": "scalar", "running_idx": "scalar"},
                    algorithm="online_argmin",
                    output_shape=("N",),
                ),
                notes="Fused distance + argmin via online argmin. D never materialized.",
            ),
        ],
        outputs=["assignments"],
    )


# ============================================================================
# Softmax (FlashAttention-style)
# ============================================================================

def softmax_baseline() -> ComputationGraph:
    """Standard softmax: materialize intermediate S = QK^T."""
    return ComputationGraph(
        name="softmax_standard",
        description="Standard attention softmax with S materialization",
        inputs={
            "Q": TensorSpec(shape=("N", "d"), dtype="f32"),
            "K_mat": TensorSpec(shape=("N", "d"), dtype="f32"),
        },
        operations=[
            Op(name="compute_S",
               reads=["Q", "K_mat"],
               computes="matmul",
               flops_fn=lambda p: 2 * p["N"] * p["N"] * p["d"],
               output=TensorSpec(shape=("N", "N"), dtype="f32", storage="HBM"),
               output_name="S",
               notes="Materialize N×N attention score matrix"),

            Op(name="softmax",
               reads=["S"],
               computes="row_softmax",
               flops_fn=lambda p: 5 * p["N"] * p["N"],
               output=TensorSpec(shape=("N", "N"), dtype="f32", storage="HBM"),
               output_name="P"),
        ],
        outputs=["P"],
    )


def softmax_flash() -> ComputationGraph:
    """Flash softmax: online softmax, no S materialization."""
    return ComputationGraph(
        name="softmax_flash",
        description="Flash attention softmax: online softmax, zero S materialization",
        inputs={
            "Q": TensorSpec(shape=("N", "d"), dtype="f32"),
            "K_mat": TensorSpec(shape=("N", "d"), dtype="f32"),
        },
        operations=[
            FusedOp(
                name="flash_softmax",
                reads=["Q", "K_mat"],
                computes="matmul+row_softmax[online_softmax]",
                flops_fn=lambda p: 2 * p["N"] * p["N"] * p["d"] + 5 * p["N"] * p["N"],
                output=TensorSpec(shape=("N", "d"), dtype="f32", storage="HBM"),
                output_name="O",
                tiling=TilingSpec(tiles={"N": "BN"}),
                online_state=OnlineStateSpec(
                    variables={"running_max": "scalar", "running_sum_exp": "scalar", "running_o": "vector_d"},
                    algorithm="online_softmax",
                    output_shape=("N", "d"),
                ),
                notes="Fused QK^T + softmax via online softmax. S never materialized.",
            ),
        ],
        outputs=["O"],
    )


# ============================================================================
# Cross-Entropy Loss
# ============================================================================

def cross_entropy_baseline() -> ComputationGraph:
    """Standard cross-entropy: materialize logits softmax intermediate."""
    return ComputationGraph(
        name="cross_entropy_standard",
        description="Standard cross-entropy loss with intermediate materialization",
        inputs={
            "logits": TensorSpec(shape=("N", "V"), dtype="f32"),
            "labels": TensorSpec(shape=("N",), dtype="i32"),
        },
        operations=[
            Op(name="row_max",
               reads=["logits"],
               computes="row_reduce_max",
               flops_fn=lambda p: p["N"] * p["V"],
               output=TensorSpec(shape=("N",), dtype="f32", storage="HBM"),
               output_name="max_vals"),

            Op(name="subtract_and_exp",
               reads=["logits", "max_vals"],
               computes="elementwise",
               flops_fn=lambda p: 2 * p["N"] * p["V"],
               output=TensorSpec(shape=("N", "V"), dtype="f32", storage="HBM"),
               output_name="exp_logits",
               notes="Materialize exp(logits - max) to HBM"),

            Op(name="sum_and_log",
               reads=["exp_logits"],
               computes="row_reduce_sum+log",
               flops_fn=lambda p: p["N"] * p["V"] + p["N"],
               output=TensorSpec(shape=("N",), dtype="f32", storage="HBM"),
               output_name="log_sum_exp"),

            Op(name="gather_loss",
               reads=["logits", "max_vals", "log_sum_exp", "labels"],
               computes="gather+subtract",
               flops_fn=lambda p: 3 * p["N"],
               output=TensorSpec(shape=("N",), dtype="f32", storage="HBM"),
               output_name="loss"),
        ],
        outputs=["loss"],
    )


def cross_entropy_flash() -> ComputationGraph:
    """Flash cross-entropy: fused online logsumexp, no intermediate materialization."""
    return ComputationGraph(
        name="cross_entropy_flash",
        description="Flash cross-entropy: fused online logsumexp over vocab dim",
        inputs={
            "logits": TensorSpec(shape=("N", "V"), dtype="f32"),
            "labels": TensorSpec(shape=("N",), dtype="i32"),
        },
        operations=[
            FusedOp(
                name="flash_cross_entropy",
                reads=["logits", "labels"],
                computes="online_logsumexp+gather",
                flops_fn=lambda p: p["N"] * p["V"] * 3 + 3 * p["N"],
                output=TensorSpec(shape=("N",), dtype="f32", storage="HBM"),
                output_name="loss",
                tiling=TilingSpec(tiles={"V": "BV"}),
                online_state=OnlineStateSpec(
                    variables={"running_max": "scalar", "running_sum_exp": "scalar"},
                    algorithm="online_logsumexp",
                    output_shape=("N",),
                ),
                notes="Fused logsumexp + gather. exp_logits never materialized.",
            ),
        ],
        outputs=["loss"],
    )


# Registry of all examples
EXAMPLES = {
    "gmm_estep": {
        "baseline": gmm_estep_baseline,
        "flash": gmm_estep_flash,
        "default_params": {"N": 65536, "K": 1024, "d": 128, "BN": 64, "BK": 8},
    },
    "gmm_em_fused": {
        "baseline": gmm_estep_baseline,
        "flash": gmm_em_fused_flash,
        "default_params": {"N": 65536, "K": 1024, "d": 128, "BN": 64, "BK": 8},
    },
    "kmeans": {
        "baseline": kmeans_assignment_baseline,
        "flash": kmeans_assignment_flash,
        "default_params": {"N": 65536, "K": 1024, "d": 128, "BN": 64, "BK": 8},
    },
    "softmax": {
        "baseline": softmax_baseline,
        "flash": softmax_flash,
        "default_params": {"N": 4096, "d": 128, "BN": 64},
    },
    "cross_entropy": {
        "baseline": cross_entropy_baseline,
        "flash": cross_entropy_flash,
        "default_params": {"N": 4096, "V": 32000, "BV": 1024},
    },
}
