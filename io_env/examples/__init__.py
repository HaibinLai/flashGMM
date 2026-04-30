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
        outputs=["log_norm"],
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


# ============================================================================
# LayerNorm
# ============================================================================

def layernorm_baseline() -> ComputationGraph:
    """Standard LayerNorm: mean → variance → normalize. Materializes intermediate stats."""
    return ComputationGraph(
        name="layernorm_standard",
        description="Standard LayerNorm with materialized mean and variance intermediates",
        inputs={
            "X": TensorSpec(shape=("N", "d"), dtype="f32"),
            "gamma": TensorSpec(shape=("d",), dtype="f32"),
            "beta": TensorSpec(shape=("d",), dtype="f32"),
        },
        operations=[
            Op(name="compute_mean",
               reads=["X"],
               computes="row_reduce_mean",
               flops_fn=lambda p: p["N"] * p["d"],
               output=TensorSpec(shape=("N",), dtype="f32", storage="HBM"),
               output_name="mean",
               notes="Materialize per-row mean"),

            Op(name="compute_var",
               reads=["X", "mean"],
               computes="row_reduce_variance",
               flops_fn=lambda p: p["N"] * p["d"] * 3,
               output=TensorSpec(shape=("N",), dtype="f32", storage="HBM"),
               output_name="var",
               notes="Materialize per-row variance (reads X again!)"),

            Op(name="normalize",
               reads=["X", "mean", "var", "gamma", "beta"],
               computes="elementwise_affine",
               flops_fn=lambda p: p["N"] * p["d"] * 5,
               output=TensorSpec(shape=("N", "d"), dtype="f32", storage="HBM"),
               output_name="Y"),
        ],
        outputs=["Y"],
    )


def layernorm_flash() -> ComputationGraph:
    """Flash LayerNorm: single-pass Welford + normalize."""
    return ComputationGraph(
        name="layernorm_flash",
        description="Flash LayerNorm: single-pass Welford mean+var, fused normalize",
        inputs={
            "X": TensorSpec(shape=("N", "d"), dtype="f32"),
            "gamma": TensorSpec(shape=("d",), dtype="f32"),
            "beta": TensorSpec(shape=("d",), dtype="f32"),
        },
        operations=[
            FusedOp(
                name="flash_layernorm",
                reads=["X", "gamma", "beta"],
                computes="welford_mean_var+normalize[online_welford]",
                flops_fn=lambda p: p["N"] * p["d"] * 9,
                output=TensorSpec(shape=("N", "d"), dtype="f32", storage="HBM"),
                output_name="Y",
                online_state=OnlineStateSpec(
                    variables={"running_mean": "scalar", "running_M2": "scalar", "count": "scalar"},
                    algorithm="online_welford",
                    output_shape=("N", "d"),
                ),
                notes="2-pass: Welford mean+var (1 read) + normalize (1 read, 1 write). No mean/var materialized.",
            ),
        ],
        outputs=["Y"],
    )


# ============================================================================
# Cosine Similarity (pairwise)
# ============================================================================

def cosine_similarity_baseline() -> ComputationGraph:
    """Pairwise cosine similarity: normalize → matmul. Materializes norms and normalized matrix."""
    return ComputationGraph(
        name="cosine_similarity_standard",
        description="Pairwise cosine similarity with materialized norms and normalized matrix",
        inputs={
            "X": TensorSpec(shape=("N", "d"), dtype="f32"),
        },
        operations=[
            Op(name="compute_norms",
               reads=["X"],
               computes="row_reduce_l2norm",
               flops_fn=lambda p: p["N"] * p["d"] * 2,
               output=TensorSpec(shape=("N",), dtype="f32", storage="HBM"),
               output_name="norms",
               notes="Materialize L2 norms"),

            Op(name="normalize_rows",
               reads=["X", "norms"],
               computes="elementwise_div",
               flops_fn=lambda p: p["N"] * p["d"],
               output=TensorSpec(shape=("N", "d"), dtype="f32", storage="HBM"),
               output_name="X_normed",
               notes="Materialize normalized X (same size as input!)"),

            Op(name="matmul_similarity",
               reads=["X_normed"],
               computes="matmul_self",
               flops_fn=lambda p: 2 * p["N"] * p["N"] * p["d"],
               output=TensorSpec(shape=("N", "N"), dtype="f32", storage="HBM"),
               output_name="S"),
        ],
        outputs=["S"],
    )


# ============================================================================
# Contrastive Loss (InfoNCE / SimCLR style)
# ============================================================================

def contrastive_loss_baseline() -> ComputationGraph:
    """InfoNCE contrastive loss: similarity → logsumexp → gather. Huge N×N intermediate."""
    return ComputationGraph(
        name="contrastive_loss_standard",
        description="InfoNCE contrastive loss with materialized N×N similarity matrix",
        inputs={
            "Z": TensorSpec(shape=("N", "d"), dtype="f32"),
            "labels": TensorSpec(shape=("N",), dtype="i32"),
            "temperature": TensorSpec(shape=(1,), dtype="f32"),
        },
        operations=[
            Op(name="compute_norms",
               reads=["Z"],
               computes="row_reduce_l2norm",
               flops_fn=lambda p: p["N"] * p["d"] * 2,
               output=TensorSpec(shape=("N",), dtype="f32", storage="HBM"),
               output_name="norms"),

            Op(name="compute_similarity",
               reads=["Z", "norms", "temperature"],
               computes="normalized_matmul",
               flops_fn=lambda p: 2 * p["N"] * p["N"] * p["d"] + p["N"] * p["N"],
               output=TensorSpec(shape=("N", "N"), dtype="f32", storage="HBM"),
               output_name="sim_matrix",
               notes="Materialize HUGE N×N similarity matrix!"),

            Op(name="logsumexp_rows",
               reads=["sim_matrix"],
               computes="row_reduce_logsumexp",
               flops_fn=lambda p: p["N"] * p["N"] * 3,
               output=TensorSpec(shape=("N",), dtype="f32", storage="HBM"),
               output_name="log_denom"),

            Op(name="gather_positive",
               reads=["sim_matrix", "labels", "log_denom"],
               computes="gather_subtract",
               flops_fn=lambda p: p["N"] * 3,
               output=TensorSpec(shape=("N",), dtype="f32", storage="HBM"),
               output_name="loss"),
        ],
        outputs=["loss"],
    )


# Update registry
EXAMPLES.update({
    "layernorm": {
        "baseline": layernorm_baseline,
        "flash": layernorm_flash,
        "default_params": {"N": 16384, "d": 4096, "BN": 64},
    },
    "cosine_similarity": {
        "baseline": cosine_similarity_baseline,
        "flash": cosine_similarity_baseline,  # no flash version yet — let agent discover it
        "default_params": {"N": 4096, "d": 256, "BN": 64},
    },
    "contrastive_loss": {
        "baseline": contrastive_loss_baseline,
        "flash": contrastive_loss_baseline,  # no flash version — agent must discover it
        "default_params": {"N": 4096, "d": 128, "BN": 64},
    },
})


# ============================================================================
# 1D FFT (radix-2 Cooley-Tukey)
# ============================================================================

def fft_baseline() -> ComputationGraph:
    """
    Standard 1D FFT: log2(N) butterfly passes, each materializing a full N-length
    complex buffer to HBM.  Total IO = O(N * log N) reads + writes.
    """
    return ComputationGraph(
        name="fft_standard",
        description="Radix-2 Cooley-Tukey FFT with per-stage HBM materialization",
        inputs={
            "x_re": TensorSpec(shape=("N",), dtype="f32"),
            "x_im": TensorSpec(shape=("N",), dtype="f32"),
        },
        operations=[
            Op(name="bit_reversal",
               reads=["x_re", "x_im"],
               computes="permutation",
               flops_fn=lambda p: p["N"],
               output=TensorSpec(shape=("N",), dtype="f32", storage="HBM"),
               output_name="buf_re",
               notes="Bit-reversal permutation, write reordered buffer to HBM"),

            Op(name="butterfly_passes",
               reads=["buf_re"],
               computes="butterfly_log_n",
               # log2(N) passes, each reading+writing 2N floats (re+im)
               # Model total IO as writing log2N copies of the buffer
               flops_fn=lambda p: int(p["N"] * p["log2N"] * 10),
               output=TensorSpec(shape=("log2N", "N"), dtype="f32", storage="HBM"),
               output_name="stage_buffers",
               notes="log2(N) butterfly passes; each pass reads/writes full N-complex buffer to HBM. "
                     "Total HBM IO: log2(N) × 2 × N × 4 bytes."),

            Op(name="extract_output",
               reads=["stage_buffers"],
               computes="copy",
               flops_fn=lambda p: p["N"],
               output=TensorSpec(shape=("N",), dtype="f32", storage="HBM"),
               output_name="Y_re"),
        ],
        outputs=["Y_re"],
    )


def fft_flash() -> ComputationGraph:
    """
    Flash FFT: fuse log2(BN) butterfly stages within SMEM tile.
    Each tile processes log2(BN) stages on-chip before writing back.
    Only (log2N - log2BN) global stages need HBM round-trips.
    """
    return ComputationGraph(
        name="fft_flash",
        description="Flash FFT: fuse log2(BN) butterfly stages in SMEM per tile",
        inputs={
            "x_re": TensorSpec(shape=("N",), dtype="f32"),
            "x_im": TensorSpec(shape=("N",), dtype="f32"),
        },
        operations=[
            FusedOp(
                name="flash_fft",
                reads=["x_re", "x_im"],
                computes="bit_reversal+butterfly[fused_stages]",
                flops_fn=lambda p: int(p["N"] * p["log2N"] * 10),
                output=TensorSpec(shape=("N",), dtype="f32", storage="HBM"),
                output_name="Y_re",
                tiling=TilingSpec(tiles={"N": "BN"}),
                notes="Fuse log2(BN)=10 stages in SMEM. Only (log2N - log2BN) = 10 global stages "
                      "need HBM round-trips (vs 20 for standard). ~50% IO reduction.",
            ),
        ],
        outputs=["Y_re"],
    )


# ============================================================================
# Conv2D (im2col + GEMM approach)
# ============================================================================

def conv2d_baseline() -> ComputationGraph:
    """
    Standard Conv2D via im2col: unfold input → GEMM → output.
    im2col materializes a huge (N*OH*OW, C_in*KH*KW) matrix.
    """
    return ComputationGraph(
        name="conv2d_standard",
        description="Conv2D via im2col + GEMM with materialized unfolded matrix",
        inputs={
            "input": TensorSpec(shape=("N", "C_in", "H", "W"), dtype="f32"),
            "weight": TensorSpec(shape=("C_out", "C_in", "KH", "KW"), dtype="f32"),
        },
        operations=[
            Op(name="im2col",
               reads=["input"],
               computes="unfold",
               flops_fn=lambda p: p["N"] * p["OH"] * p["OW"] * p["C_in"] * p["KH"] * p["KW"],
               output=TensorSpec(shape=("N_OH_OW", "C_in_KH_KW"), dtype="f32", storage="HBM"),
               output_name="col_matrix",
               notes="Materialize unfolded matrix: (N*OH*OW) × (C_in*KH*KW). "
                     "For N=64, C=128, H=W=56, K=3: 12.8M × 1152 = 56 GB!"),

            Op(name="gemm",
               reads=["col_matrix", "weight"],
               computes="matmul",
               flops_fn=lambda p: 2 * p["N"] * p["OH"] * p["OW"] * p["C_out"] * p["C_in"] * p["KH"] * p["KW"],
               output=TensorSpec(shape=("N", "C_out", "OH", "OW"), dtype="f32", storage="HBM"),
               output_name="output",
               notes="GEMM on unfolded matrix"),
        ],
        outputs=["output"],
    )


def conv2d_flash() -> ComputationGraph:
    """
    Flash Conv2D: implicit im2col fused with GEMM.
    Compute im2col on-the-fly in SMEM, never materialize to HBM.
    """
    return ComputationGraph(
        name="conv2d_flash",
        description="Flash Conv2D: implicit im2col fused with GEMM, zero unfolded matrix",
        inputs={
            "input": TensorSpec(shape=("N", "C_in", "H", "W"), dtype="f32"),
            "weight": TensorSpec(shape=("C_out", "C_in", "KH", "KW"), dtype="f32"),
        },
        operations=[
            FusedOp(
                name="flash_conv2d",
                reads=["input", "weight"],
                computes="implicit_im2col+gemm",
                flops_fn=lambda p: 2 * p["N"] * p["OH"] * p["OW"] * p["C_out"] * p["C_in"] * p["KH"] * p["KW"],
                output=TensorSpec(shape=("N", "C_out", "OH", "OW"), dtype="f32", storage="HBM"),
                output_name="output",
                tiling=TilingSpec(tiles={"N_OH_OW": "BM", "C_out": "BN", "C_in_KH_KW": "BK"}),
                notes="Implicit im2col: gather input patches into SMEM on-the-fly, "
                      "then GEMM tile against weight. col_matrix never in HBM.",
            ),
        ],
        outputs=["output"],
    )


# ============================================================================
# 2D Stencil (Jacobi iteration)
# ============================================================================

def stencil2d_baseline() -> ComputationGraph:
    """
    Standard 2D 5-point Jacobi stencil: each of T iterations reads H×W grid
    and writes H×W output to HBM. Total IO = T × 2 × H × W × 4 bytes.
    Model this as: grid → T iterations of intermediate buffers → grid_out.
    """
    return ComputationGraph(
        name="stencil2d_standard",
        description="2D Jacobi stencil: T iterations, each materializes full H×W grid",
        inputs={
            "grid": TensorSpec(shape=("H", "W"), dtype="f32"),
        },
        operations=[
            Op(name="stencil_iterations",
               reads=["grid"],
               computes="jacobi_5pt",
               # T iterations, each doing 5 FLOPs per cell
               flops_fn=lambda p: p["T"] * p["H"] * p["W"] * 5,
               # Model total materialization: T copies of H×W (T HBM round-trips)
               output=TensorSpec(shape=("T", "H", "W"), dtype="f32", storage="HBM"),
               output_name="iter_buffers",
               notes="T iterations, each reads + writes full H×W grid. "
                     "Intermediate shape (T,H,W) models total materialization."),

            Op(name="extract_final",
               reads=["iter_buffers"],
               computes="copy",
               flops_fn=lambda p: p["H"] * p["W"],
               output=TensorSpec(shape=("H", "W"), dtype="f32", storage="HBM"),
               output_name="grid_out"),
        ],
        outputs=["grid_out"],
    )


def stencil2d_flash() -> ComputationGraph:
    """
    Flash 2D Stencil: temporal tiling — fuse S iterations within SMEM tile.
    Each tile loads (BH+2S) × (BW+2S) halo region once, applies S iterations
    on-chip, writes back BH×BW output.
    Total HBM IO: ceil(T/S) × H×W + halo overhead, instead of T × H×W.
    """
    return ComputationGraph(
        name="stencil2d_flash",
        description="Flash stencil: temporal tiling, fuse S iterations in SMEM",
        inputs={
            "grid": TensorSpec(shape=("H", "W"), dtype="f32"),
        },
        operations=[
            FusedOp(
                name="flash_stencil",
                reads=["grid"],
                computes="jacobi_5pt[temporal_tiling]",
                flops_fn=lambda p: p["T"] * p["H"] * p["W"] * 5,
                output=TensorSpec(shape=("H", "W"), dtype="f32", storage="HBM"),
                output_name="grid_out",
                tiling=TilingSpec(tiles={"H": "BH", "W": "BW", "T": "S"}),
                notes="Temporal tiling: fuse S iterations per tile. "
                      "IO: ceil(T/S) × H×W reads + writes (S× fewer HBM round-trips). "
                      "No intermediate grids materialized.",
            ),
        ],
        outputs=["grid_out"],
    )


# Update registry with new operators
EXAMPLES.update({
    "fft": {
        "baseline": fft_baseline,
        "flash": fft_flash,
        "default_params": {"N": 1048576, "log2N": 20, "BN": 1024},
    },
    "conv2d": {
        "baseline": conv2d_baseline,
        "flash": conv2d_flash,
        "default_params": {
            "N": 64, "C_in": 128, "C_out": 256, "H": 56, "W": 56,
            "KH": 3, "KW": 3, "OH": 56, "OW": 56,
            "N_OH_OW": 64 * 56 * 56, "C_in_KH_KW": 128 * 3 * 3,
            "BM": 128, "BN": 128, "BK": 32,
        },
    },
    "stencil2d": {
        "baseline": stencil2d_baseline,
        "flash": stencil2d_flash,
        "default_params": {"H": 4096, "W": 4096, "T": 100, "BH": 64, "BW": 64, "S": 8},
    },
})
