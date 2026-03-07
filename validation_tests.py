import numpy as np
import matplotlib.pyplot as plt
from fd_solver import leapfrog_solve
from itertools import product
from scipy.interpolate import interp1d, RegularGridInterpolator


# ---------------------------------------------------------------
# Convergence Test
# ---------------------------------------------------------------
def convergence_test(d, N_list, N_ref, c=1.0, lam=1.0, T=1.0, eps=0.1):
    if d == 1:
        phi0 = lambda x: eps * np.sin(x)
    elif d == 2:
        phi0 = lambda x, y: eps * np.sin(x) * np.sin(y)
    else:
        phi0 = lambda x, y, z: eps * np.sin(x) * np.sin(y) * np.sin(z)

    print(f"Computing reference solution (N_ref={N_ref}, d={d})...")
    phi_ref, grids_ref, _, _ = leapfrog_solve(d, N_ref, c, lam, None, phi0, T)

    results = []
    for N in N_list:
        print(f"Computing N={N}...")
        phi_h, grids_h, _, _ = leapfrog_solve(d, N, c, lam, None, phi0, T)

        if d == 1:
            f_interp = interp1d(grids_ref[0], phi_ref, kind='cubic')
            phi_ref_on_coarse = f_interp(grids_h[0])
        elif d == 2:
            interp = RegularGridInterpolator(grids_ref, phi_ref, method='cubic')
            pts = np.array(list(product(*grids_h)))
            phi_ref_on_coarse = interp(pts).reshape(phi_h.shape)
        else:
            interp = RegularGridInterpolator(grids_ref, phi_ref, method='linear')
            pts = np.array(list(product(*grids_h)))
            phi_ref_on_coarse = interp(pts).reshape(phi_h.shape)

        err = phi_h - phi_ref_on_coarse
        h = np.pi / (N + 1)
        e_inf = np.max(np.abs(err))
        e_L2 = np.sqrt(h**d * np.sum(err**2))

        results.append({'N': N, 'h': h, 'e_inf': e_inf, 'e_L2': e_L2})

    for i in range(1, len(results)):
        r = results[i]
        r_prev = results[i - 1]
        r['rate_inf'] = np.log2(r_prev['e_inf'] / r['e_inf'])
        r['rate_L2'] = np.log2(r_prev['e_L2'] / r['e_L2'])
    results[0]['rate_inf'] = None
    results[0]['rate_L2'] = None

    return results


def print_table(results, d):
    print(f"\n=== d = {d} ===")
    print(f"{'N':>6} {'e_inf':>12} {'Rate':>8} {'e_L2':>12} {'Rate':>8}")
    print("-" * 50)
    for r in results:
        ri = f"{r['rate_inf']:.2f}" if r['rate_inf'] is not None else "---"
        rl = f"{r['rate_L2']:.2f}" if r['rate_L2'] is not None else "---"
        print(f"{r['N']:>6} {r['e_inf']:>12.4e} {ri:>8} {r['e_L2']:>12.4e} {rl:>8}")
    print(f"\n% LaTeX table for d={d}:")
    for r in results:
        ri = f"{r['rate_inf']:.2f}" if r['rate_inf'] is not None else "---"
        rl = f"{r['rate_L2']:.2f}" if r['rate_L2'] is not None else "---"
        print(f"{r['N']} & ${r['e_inf']:.2e}$ & {ri} & ${r['e_L2']:.2e}$ & {rl} \\\\")


# ---------------------------------------------------------------
# Linear Test (lam = 0)
# ---------------------------------------------------------------
def linear_test(N=128, T=1.0, eps=0.1):
    print("\n--- Linear Test (lam = 0) ---")
    phi0 = lambda x: eps * np.sin(x)
    phi, grids, _, _ = leapfrog_solve(1, N, c=1.0, lam=0.0,
                                       V_func=None, phi0_func=phi0, T=T)
    exact = eps * np.sin(grids[0]) * np.cos(T)
    max_err = np.max(np.abs(phi - exact))
    print(f"Max error: {max_err:.6e}")
    if max_err < 1e-4:
        print("PASS")
    return max_err


# ---------------------------------------------------------------
# CFL Violation Test
# ---------------------------------------------------------------
def cfl_test(d=1, N=128, T=1.0):
    """
    CFL violation test using the actual solver.
    N=128 gives ~27 time steps at cfl_frac=1.5, enough for the
    unstable mode to grow from roundoff (~1e-16) to blowup
    (amplification factor ~6.85^27 ~ 1e22).
    """
    print("\n--- CFL Violation Test ---")
    print(f"d={d}, N={N}, cfl_frac=1.5")
    phi0 = lambda x: 0.1 * np.sin(x)
    phi, _, _, dt = leapfrog_solve(d, N, c=1.0, lam=1.0,
                                    V_func=None, phi0_func=phi0,
                                    T=T, cfl_frac=1.5)
    h = np.pi / (N + 1)
    nu = 1.0 * dt / h
    print(f"dt = {dt:.6e}, h = {h:.6e}, nu = {nu:.3f}")
    max_val = np.max(np.abs(phi))
    if np.any(np.isnan(phi)) or max_val > 1e6:
        print(f"PASS: Blowup detected, max|phi| = {max_val:.2e}")
    else:
        print(f"WARNING: No blowup, max|phi| = {max_val:.2e}")
        print("Try N=256 or cfl_frac=2.0")


# ---------------------------------------------------------------
# Energy Conservation Test
# ---------------------------------------------------------------
def energy_test(N=256, T=1.0):
    print(f"\n--- Energy Conservation Test (N={N}) ---")
    phi0 = lambda x: 0.1 * np.sin(x)
    phi, grids, energies, dt = leapfrog_solve(1, N, c=1.0, lam=1.0,
                                               V_func=None, phi0_func=phi0,
                                               T=T, store_energy=True)
    E0 = energies[0]
    drift = np.abs((energies - E0) / E0)
    max_drift = np.max(drift)
    final_drift = drift[-1]

    print(f"dt = {dt:.6e}")
    print(f"Max |dE/E| over trajectory: {max_drift:.6e}")
    print(f"Final |dE/E|:               {final_drift:.6e}")

    if max_drift < 1e-4:
        print("PASS: energy drift < 1e-4")
    elif max_drift < 1e-3:
        print(f"MARGINAL: energy drift = {max_drift:.2e}")
    else:
        print(f"FAIL: energy drift = {max_drift:.2e} (target < 1e-4)")

    # Plot
    t = np.linspace(0, T, len(energies))
    plt.figure(figsize=(6, 4))
    plt.semilogy(t, drift)
    plt.xlabel("Time")
    plt.ylabel("|$\\Delta E / E$|")
    plt.title(f"Energy Conservation (N={N}, shadow Hamiltonian)")
    plt.tight_layout()
    plt.savefig("energy_drift.pdf")
    plt.savefig("energy_drift.png", dpi=150)
    print("Saved: energy_drift.pdf, energy_drift.png")

    return max_drift


# ---------------------------------------------------------------
# Symmetry Test (d=2)
# ---------------------------------------------------------------
def symmetry_test(N=32, T=1.0):
    print("\n--- Symmetry Test (d=2) ---")
    phi0 = lambda x, y: 0.1 * np.sin(x) * np.sin(y)
    phi, _, _, _ = leapfrog_solve(2, N, c=1.0, lam=1.0,
                                   V_func=None, phi0_func=phi0, T=T)
    asym = np.max(np.abs(phi - phi.T))
    print(f"max|Phi(x,y) - Phi(y,x)| = {asym:.2e}")
    if asym < 1e-14:
        print("PASS")
    else:
        print(f"FAIL: asymmetry = {asym:.2e}")
    return asym


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("FINITE DIFFERENCE VALIDATION SUITE")
    print("Equation: Phi_tt = c^2 Lap Phi - lam Phi^3")
    print("Domain: [0,pi]^d, Dirichlet BCs")
    print("Scheme: Leapfrog (Stormer-Verlet)")
    print("=" * 60)

    # 1. Convergence tests
    print("\n" + "=" * 40)
    print("1. CONVERGENCE TESTS")
    print("=" * 40)

    r1 = convergence_test(1, [16, 32, 64, 128, 256], N_ref=512)
    print_table(r1, 1)

    r2 = convergence_test(2, [8, 16, 32, 64], N_ref=128)
    print_table(r2, 2)

    # d=3: only report N=8,16 to avoid reference contamination at N=32
    # (N_ref=64 is only 2x finer than N=32, inflating the apparent rate)
    r3 = convergence_test(3, [8, 16], N_ref=64)
    print_table(r3, 3)

    # 2. Linear test
    print("\n" + "=" * 40)
    print("2. LINEAR TEST")
    print("=" * 40)
    linear_test()

    # 3. CFL violation test (N=128 for enough time steps)
    print("\n" + "=" * 40)
    print("3. CFL VIOLATION TEST")
    print("=" * 40)
    cfl_test(d=1, N=128)

    # 4. Energy conservation (shadow Hamiltonian)
    print("\n" + "=" * 40)
    print("4. ENERGY CONSERVATION")
    print("=" * 40)
    energy_test(N=256)
    energy_test(N=512)

    # 5. Symmetry test
    print("\n" + "=" * 40)
    print("5. SYMMETRY TEST")
    print("=" * 40)
    symmetry_test()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
