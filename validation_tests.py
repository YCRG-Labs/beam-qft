# validation_tests.py
import numpy as np
import matplotlib.pyplot as plt
from fd_solver import leapfrog_solve  # keep your original solver
from itertools import product
from scipy.interpolate import interp1d, RegularGridInterpolator

# -------------------------------
# Convergence Test
# -------------------------------
def convergence_test(d, N_list, N_ref, c=1.0, lam=1.0, T=1.0, eps=0.1):
    if d == 1:
        phi0 = lambda x: eps * np.sin(x)
    elif d == 2:
        phi0 = lambda x, y: eps * np.sin(x) * np.sin(y)
    else:
        phi0 = lambda x, y, z: eps * np.sin(x) * np.sin(y) * np.sin(z)

    print(f"Computing reference solution (N_ref={N_ref}, d={d})...")
    phi_ref, grids_ref, _ = leapfrog_solve(d, N_ref, c, lam, None, phi0, T)

    results = []
    for N in N_list:
        print(f"Computing N={N}...")
        phi_h, grids_h, _ = leapfrog_solve(d, N, c, lam, None, phi0, T)

        # Interpolate reference solution to coarse grid
        if d == 1:
            f_interp = interp1d(grids_ref[0], phi_ref, kind='cubic')
            phi_ref_on_coarse = f_interp(grids_h[0])
        else:
            interp = RegularGridInterpolator(grids_ref, phi_ref, method='cubic')
            pts = np.array(list(product(*grids_h)))
            phi_ref_on_coarse = interp(pts).reshape(phi_h.shape)

        err = phi_h - phi_ref_on_coarse
        h = np.pi / (N + 1)
        e_inf = np.max(np.abs(err))
        e_L2 = np.sqrt(h**d * np.sum(err**2))

        results.append({'N': N, 'h': h, 'e_inf': e_inf, 'e_L2': e_L2})

    # Compute rates
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
    print("-"*50)
    for r in results:
        rate_inf = f"{r['rate_inf']:.2f}" if r['rate_inf'] is not None else "---"
        rate_L2 = f"{r['rate_L2']:.2f}" if r['rate_L2'] is not None else "---"
        print(f"{r['N']:>6} {r['e_inf']:>12.4e} {rate_inf:>8} {r['e_L2']:>12.4e} {rate_L2:>8}")
    print("\n% LaTeX table for d={}".format(d))
    for r in results:
        rate_inf = f"{r['rate_inf']:.2f}" if r['rate_inf'] is not None else "---"
        rate_L2 = f"{r['rate_L2']:.2f}" if r['rate_L2'] is not None else "---"
        print(f"{r['N']} & ${r['e_inf']:.2e}$ & {rate_inf} & ${r['e_L2']:.2e}$ & {rate_L2} \\\\")

# -------------------------------
# Linear Test (λ = 0)
# -------------------------------
def linear_test_1d(N=128, T=1.0, eps=0.1):
    phi0 = lambda x: eps * np.sin(x)
    phi, grids, _ = leapfrog_solve(1, N, c=1.0, lam=0.0, V_func=None, phi0_func=phi0, T=T)
    exact = eps * np.sin(grids[0]) * np.cos(T)
    max_err = np.max(np.abs(phi - exact))
    print("\n--- Linear Test (λ = 0) ---")
    print(f"Max error: {max_err:e}")

# -------------------------------
# CFL Violation Test
# -------------------------------
def cfl_test(d=1, N=32, T=1.0):
    print("\n--- CFL Violation Test ---")
    phi0 = lambda x: 0.1 * np.sin(x)
    phi, _, _ = leapfrog_solve(d, N, c=1.0, lam=1.0,
                               V_func=None, phi0_func=phi0,
                               T=T, cfl_frac=1.1)
    max_val = np.max(np.abs(phi))
    if np.any(np.isnan(phi)) or max_val > 1e6:
        print(f"Blowup detected: max|phi| = {max_val:.2e}")
    else:
        print(f"WARNING: No blowup detected. max|phi| = {max_val:.2e}")
        print("Check that cfl_frac > 1.0 is actually exceeding the CFL limit.")

# -------------------------------
# Energy Drift Plot
# -------------------------------
def energy_drift_plot():
    phi, grids, energy_list = leapfrog_solve(d=1, N=128, c=1.0, lam=1.0,
                                             V_func=None, phi0_func=lambda x: 0.1*np.sin(x),
                                             T=1.0)
    energy_list = np.array(energy_list)
    plt.figure()
    plt.plot(np.linspace(0,1,len(energy_list)), np.abs(energy_list-energy_list[0])/energy_list[0])
    plt.xlabel("time")
    plt.ylabel("|ΔE/E|")
    plt.title("Energy Drift")
    plt.savefig("energy_drift.pdf")
    print("\nFinal energy drift:", np.abs(energy_list[-1]-energy_list[0])/energy_list[0])

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    print("="*60)
    print("FINITE DIFFERENCE VALIDATION TESTS")
    print("="*60)

    # Convergence Tests
    print("\n---- Convergence Test ----")
    r1 = convergence_test(1, [16,32,64,128,256], N_ref=512)
    print_table(r1, 1)
    r2 = convergence_test(2, [8,16,32,64], N_ref=128)
    print_table(r2, 2)
    r3 = convergence_test(3, [8,16,32], N_ref=64)
    print_table(r3, 3)

    # Linear Test
    linear_test_1d()

    # CFL Violation Test
    cfl_test()

    # Energy Drift Plot

    energy_drift_plot()
