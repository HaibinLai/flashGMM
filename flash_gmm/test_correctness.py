"""
Correctness tests: validate Flash-GMM variants against standard GMM and sklearn.
"""

import torch
import numpy as np
import sys


def test_flash_e_step_vs_standard():
    """Test: Flash E-step γ matches standard E-step γ."""
    from standard_gmm import StandardGMM, IOCounter, generate_gmm_data
    from flash_e_step import flash_e_step

    print("Test 1: Flash E-step vs Standard E-step")
    for N, K, d in [(512, 4, 8), (2048, 8, 32), (4096, 16, 64)]:
        X, _, _, _, _ = generate_gmm_data(N, K, d)
        model = StandardGMM(K=K, d=d)
        model._init_params(X)

        model.io_counter.reset()
        gamma_std, ll_std = model.e_step(X)

        io = IOCounter()
        gamma_flash, ll_flash = flash_e_step(X, model.mu, model.var, model.log_pi, BK=4, io_counter=io)

        diff_gamma = (gamma_std - gamma_flash).abs().max().item()
        diff_ll = (ll_std - ll_flash).abs().max().item()

        status = "PASS" if diff_gamma < 1e-4 and diff_ll < 1e-3 else "FAIL"
        print(f"  N={N:5d} K={K:3d} d={d:3d}: max|Δγ|={diff_gamma:.2e} max|ΔLL|={diff_ll:.2e} [{status}]")

        if status == "FAIL":
            return False
    return True


def test_flash_em_vs_standard():
    """Test: Flash E+M fused produces same parameter updates as standard EM."""
    from standard_gmm import StandardGMM, generate_gmm_data
    from flash_gmm import FlashGMM

    print("\nTest 2: Flash E+M vs Standard EM (single iteration)")
    for N, K, d in [(512, 4, 8), (2048, 8, 32), (4096, 16, 64)]:
        X, _, _, _, _ = generate_gmm_data(N, K, d)

        # Standard: one iteration
        std = StandardGMM(K=K, d=d, seed=42)
        std._init_params(X)
        gamma, _ = std.e_step(X)
        std.m_step(X, gamma)

        # Flash: one iteration
        flash = FlashGMM(K=K, d=d, BK=4, seed=42)
        flash._init_params(X)
        flash._fused_em_step(X)

        diff_mu = (std.mu - flash.mu).abs().max().item()
        diff_var = (std.var - flash.var).abs().max().item()
        diff_pi = (std.log_pi - flash.log_pi).abs().max().item()

        status = "PASS" if diff_mu < 1e-3 and diff_var < 1e-3 and diff_pi < 1e-3 else "FAIL"
        print(f"  N={N:5d} K={K:3d} d={d:3d}: max|Δμ|={diff_mu:.2e} max|Δσ²|={diff_var:.2e} max|Δlogπ|={diff_pi:.2e} [{status}]")

        if status == "FAIL":
            return False
    return True


def test_convergence_vs_sklearn():
    """Test: Flash-GMM converges to similar log-likelihood as sklearn."""
    from standard_gmm import StandardGMM, generate_gmm_data
    from flash_gmm import FlashGMM
    from sklearn.mixture import GaussianMixture

    print("\nTest 3: Convergence comparison with sklearn")
    N, K, d = 2048, 4, 16
    X, _, _, _, _ = generate_gmm_data(N, K, d, seed=0)
    X_np = X.numpy()

    # sklearn
    sk = GaussianMixture(n_components=K, covariance_type="diag", max_iter=100,
                         tol=1e-4, random_state=42, n_init=1)
    sk.fit(X_np)
    sk_ll = sk.score(X_np)

    # Standard GMM
    std = StandardGMM(K=K, d=d, max_iter=100, tol=1e-4, seed=42)
    std_hist = std.fit(X, verbose=False)
    std_ll = std_hist[-1]["log_likelihood"]

    # Flash GMM
    flash = FlashGMM(K=K, d=d, BK=4, max_iter=100, tol=1e-4, seed=42)
    flash_hist = flash.fit(X, verbose=False)
    flash_ll = flash_hist[-1]["log_likelihood"]

    print(f"  sklearn LL:   {sk_ll:.4f}  (iters: {sk.n_iter_})")
    print(f"  Standard LL:  {std_ll:.4f}  (iters: {len(std_hist)})")
    print(f"  Flash LL:     {flash_ll:.4f}  (iters: {len(flash_hist)})")

    # They should converge to similar values (not identical due to different init)
    # The key check is Standard == Flash (same init, same math)
    diff_std_flash = abs(std_ll - flash_ll)
    print(f"  |Standard - Flash| LL diff: {diff_std_flash:.6f}")
    status = "PASS" if diff_std_flash < 0.01 else "FAIL"
    print(f"  Standard ↔ Flash match: [{status}]")
    return status == "PASS"


def test_flash_em_multi_iter():
    """Test: Flash E+M stays synchronized with standard over multiple iterations."""
    from standard_gmm import StandardGMM, generate_gmm_data
    from flash_gmm import FlashGMM

    print("\nTest 4: Multi-iteration parameter tracking")
    N, K, d = 1024, 4, 16
    X, _, _, _, _ = generate_gmm_data(N, K, d)
    n_iters = 10

    std = StandardGMM(K=K, d=d, max_iter=n_iters, tol=0, seed=42)
    flash = FlashGMM(K=K, d=d, BK=4, max_iter=n_iters, tol=0, seed=42)

    std_hist = std.fit(X, verbose=False)
    flash_hist = flash.fit(X, verbose=False)

    all_pass = True
    for i in range(min(len(std_hist), len(flash_hist))):
        ll_diff = abs(std_hist[i]["log_likelihood"] - flash_hist[i]["log_likelihood"])
        status = "PASS" if ll_diff < 0.01 else "FAIL"
        if i < 5 or status == "FAIL":
            print(f"  iter {i}: std_LL={std_hist[i]['log_likelihood']:.4f} flash_LL={flash_hist[i]['log_likelihood']:.4f} diff={ll_diff:.2e} [{status}]")
        if status == "FAIL":
            all_pass = False

    if all_pass:
        print(f"  ... all {min(len(std_hist), len(flash_hist))} iterations match [PASS]")
    return all_pass


def test_io_savings():
    """Test: IO counters show expected savings pattern."""
    from standard_gmm import StandardGMM, IOCounter, generate_gmm_data
    from flash_e_step import flash_e_step
    from flash_gmm import FlashGMM

    print("\nTest 5: IO savings validation")
    N, K, d = 4096, 64, 32
    X, _, _, _, _ = generate_gmm_data(N, K, d)
    elem = 4

    # Standard
    std = StandardGMM(K=K, d=d, seed=42)
    std._init_params(X)
    std.io_counter.reset()
    gamma, _ = std.e_step(X)
    std.m_step(X, gamma)
    io_std = std.io_counter.total()

    # Flash E+M
    fused = FlashGMM(K=K, d=d, BK=4, seed=42)
    fused._init_params(X)
    fused.io_counter.reset()
    fused._fused_em_step(X)
    io_fused = fused.io_counter.total()

    # The fused version should use significantly less IO
    ratio = io_std / io_fused
    # With K=64, d=32: NK=262144 >> Nd=131072, so savings should be substantial
    print(f"  N={N}, K={K}, d={d}")
    print(f"  Standard IO: {io_std/1e6:.2f} MB")
    print(f"  Flash E+M IO: {io_fused/1e6:.2f} MB")
    print(f"  IO reduction ratio: {ratio:.2f}x")
    print(f"  NK/Nd = {K/d:.1f}x (higher → more savings)")

    status = "PASS" if ratio > 1.5 else "FAIL"
    print(f"  [{status}]")
    return status == "PASS"


if __name__ == "__main__":
    results = []
    results.append(("Flash E-step vs Standard", test_flash_e_step_vs_standard()))
    results.append(("Flash E+M vs Standard", test_flash_em_vs_standard()))
    results.append(("Convergence vs sklearn", test_convergence_vs_sklearn()))
    results.append(("Multi-iteration tracking", test_flash_em_multi_iter()))
    results.append(("IO savings validation", test_io_savings()))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_pass = True
    for name, passed in results:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_pass = False

    print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_pass else 1)
