# Numerical Validation Plan for the Dense Beam QFT Paper

## Overview

This document specifies all numerical experiments needed to fill the blank appendices (E, F, G) and validate the theoretical results in the paper. Everything is implementable in Python with NumPy/SciPy. No external physics libraries are needed.

The work divides into three independent modules that can be developed in parallel.

---

## Module 1: Finite Difference Convergence (Appendix E)

### 1.1 The PDE

The nonlinear wave equation on $\Omega = [0, \pi]^d$ with Dirichlet boundary conditions:

```
∂²Φ/∂t² = c² ΔΦ - λ Φ³ + V(x) Φ
```

Parameters for all tests:
- `c = 1.0` (wave speed)
- `lambda_ = 1.0` (nonlinear coupling, defocusing)
- `V = 0.0` (no external potential — simplest case; run V≠0 as a bonus)
- `T = 1.0` (final time)
- Initial data: `Φ₀(x) = ε * ∏ sin(xⱼ)` with `ε = 0.1`
- Initial velocity: `∂Φ/∂t(x,0) = 0`

### 1.2 The Scheme (Leapfrog / Störmer-Verlet)

```python
# Time stepping:
# Φ^{n+1} = 2*Φ^n - Φ^{n-1} + dt² * (c² * Δ_h Φ^n + F(Φ^n))
#
# where Δ_h is the standard (2d+1)-point discrete Laplacian:
# (Δ_h Φ)_i = (1/h²) * sum_j [Φ_{i+e_j} - 2*Φ_i + Φ_{i-e_j}]
#
# and F(Φ) = -λ Φ³ + V Φ
```

CFL condition: `ν = c * dt / h ≤ 1/√d`. Use `ν = 0.9/√d` (90% of the stability limit).

For the first time step (`n=0 → n=1`), use the Taylor expansion:
```
Φ¹ = Φ⁰ + dt * Φ̇⁰ + (dt²/2) * (c² Δ_h Φ⁰ + F(Φ⁰))
```
Since `Φ̇⁰ = 0`, this simplifies to `Φ¹ = Φ⁰ + (dt²/2) * RHS(Φ⁰)`.

### 1.3 Tests to Run

**d = 1:**
| Grid | N_ref | N_coarse | Expected |
|------|-------|----------|----------|
| 1D | 512 | 16, 32, 64, 128, 256 | Rate ≈ 2.00 |

**d = 2:**
| Grid | N_ref | N_coarse | Expected |
|------|-------|----------|----------|
| 2D | 128 | 8, 16, 32, 64 | Rate ≈ 2.00 |

**d = 3:**
| Grid | N_ref | N_coarse | Expected |
|------|-------|----------|----------|
| 3D | 64 | 8, 16, 32 | Rate ≈ 2.00 |

For each test, compute:
- `e_inf = max |Φ_h - Φ_ref|` (L∞ error, interpolating reference to coarse grid or vice versa)
- `e_L2 = sqrt(h^d * sum |Φ_h - Φ_ref|²)` (L² error)
- `rate = log2(e_{h} / e_{h/2})` (convergence rate between successive refinements)

### 1.4 Implementation Skeleton

```python
import numpy as np
from itertools import product

def leapfrog_solve(d, N, c, lam, V_func, phi0_func, T, cfl_frac=0.9):
    """
    Solve ∂²Φ/∂t² = c² ΔΦ - λ Φ³ + V(x) Φ on [0,π]^d
    with Dirichlet BCs using the leapfrog scheme.
    
    Parameters
    ----------
    d : int
        Spatial dimension (1, 2, or 3).
    N : int
        Number of interior grid points per dimension.
    c : float
        Wave speed.
    lam : float
        Nonlinear coupling (λ > 0 for defocusing).
    V_func : callable or None
        External potential V(x). Takes d-dim array, returns scalar array.
    phi0_func : callable
        Initial data Φ₀(x). Takes d-dim coordinate arrays.
    T : float
        Final time.
    cfl_frac : float
        Fraction of CFL limit to use (default 0.9).
    
    Returns
    -------
    phi : ndarray, shape (N,)*d
        Solution at time T on interior grid points.
    x_grids : tuple of 1d arrays
        Grid coordinates in each dimension.
    """
    h = np.pi / (N + 1)
    dt = cfl_frac * h / (c * np.sqrt(d))
    Nt = int(np.ceil(T / dt))
    dt = T / Nt  # adjust to hit T exactly
    
    # Grid coordinates (interior points only)
    x1d = np.linspace(h, np.pi - h, N)
    
    if d == 1:
        X = x1d
        phi = phi0_func(X)
    elif d == 2:
        X, Y = np.meshgrid(x1d, x1d, indexing='ij')
        phi = phi0_func(X, Y)
    elif d == 3:
        X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')
        phi = phi0_func(X, Y, Z)
    
    # Precompute V on grid
    if V_func is not None:
        if d == 1: V = V_func(X)
        elif d == 2: V = V_func(X, Y)
        else: V = V_func(X, Y, Z)
    else:
        V = 0.0
    
    def laplacian(u):
        """Standard (2d+1)-point discrete Laplacian with zero BCs."""
        lap = -2.0 * d * u / h**2
        for axis in range(d):
            lap += (np.roll(u, 1, axis=axis) + np.roll(u, -1, axis=axis)) / h**2
            # Fix boundary: rolled-in values should be 0 (Dirichlet)
            # For axis j, the first and last slices along axis j
            # picked up wrong values from np.roll. Fix them:
            slc_lo = [slice(None)] * d
            slc_hi = [slice(None)] * d
            slc_lo[axis] = 0
            slc_hi[axis] = -1
            # The roll brought in the opposite boundary value; subtract it
            lap[tuple(slc_lo)] -= np.roll(u, 1, axis=axis)[tuple(slc_lo)] / h**2
            lap[tuple(slc_hi)] -= np.roll(u, -1, axis=axis)[tuple(slc_hi)] / h**2
        return lap
    
    # NOTE: The np.roll approach for Dirichlet BCs is tricky.
    # A cleaner approach pads with zeros:
    def laplacian_clean(u):
        """Discrete Laplacian with zero-padded Dirichlet BCs."""
        shape_padded = tuple(s + 2 for s in u.shape)
        u_pad = np.zeros(shape_padded)
        interior = tuple(slice(1, -1) for _ in range(d))
        u_pad[interior] = u
        
        lap = -2.0 * d * u / h**2
        for axis in range(d):
            slc_p = [slice(1, -1)] * d
            slc_m = [slice(1, -1)] * d
            slc_p[axis] = slice(2, None)
            slc_m[axis] = slice(0, -2)
            lap += (u_pad[tuple(slc_p)] + u_pad[tuple(slc_m)]) / h**2
        return lap
    
    def rhs(u):
        return c**2 * laplacian_clean(u) - lam * u**3 + V * u
    
    # First step: Φ¹ = Φ⁰ + (dt²/2) * RHS(Φ⁰)  [since Φ̇⁰ = 0]
    phi_prev = phi.copy()
    phi_curr = phi + 0.5 * dt**2 * rhs(phi)
    
    # Leapfrog iteration
    for n in range(1, Nt):
        phi_next = 2.0 * phi_curr - phi_prev + dt**2 * rhs(phi_curr)
        phi_prev = phi_curr
        phi_curr = phi_next
    
    if d == 1:
        return phi_curr, (x1d,)
    elif d == 2:
        return phi_curr, (x1d, x1d)
    else:
        return phi_curr, (x1d, x1d, x1d)


def convergence_test(d, N_list, N_ref, c=1.0, lam=1.0, T=1.0, eps=0.1):
    """
    Run convergence test in d dimensions.
    
    Parameters
    ----------
    d : int
        Dimension.
    N_list : list of int
        Coarse grid sizes to test.
    N_ref : int
        Reference (fine) grid size.
    
    Returns
    -------
    results : list of dict
        Each dict has keys: N, h, e_inf, e_L2, rate_inf, rate_L2
    """
    # Initial data
    if d == 1:
        phi0 = lambda x: eps * np.sin(x)
    elif d == 2:
        phi0 = lambda x, y: eps * np.sin(x) * np.sin(y)
    else:
        phi0 = lambda x, y, z: eps * np.sin(x) * np.sin(y) * np.sin(z)
    
    # Reference solution
    print(f"  Computing reference solution (N_ref={N_ref}, d={d})...")
    phi_ref, grids_ref = leapfrog_solve(d, N_ref, c, lam, None, phi0, T)
    
    results = []
    for N in N_list:
        print(f"  Computing N={N}...")
        phi_h, grids_h = leapfrog_solve(d, N, c, lam, None, phi0, T)
        
        # Interpolate: subsample reference solution at coarse grid points
        # Since both grids are uniform on [0,π], coarse points are a subset
        # of fine points when N_ref / N is integer. Otherwise, interpolate.
        step = N_ref // N  # assumes N_ref is divisible by N (design grids this way)
        
        if d == 1:
            # Subsample: take every `step`-th point, offset to align
            # Fine grid: h_ref = π/(N_ref+1), points at h_ref, 2h_ref, ...
            # Coarse grid: h = π/(N+1), points at h, 2h, ...
            # These don't exactly coincide unless (N+1) divides (N_ref+1).
            # Safest: use scipy interpolation.
            from scipy.interpolate import interp1d
            f_interp = interp1d(grids_ref[0], phi_ref, kind='cubic')
            phi_ref_on_coarse = f_interp(grids_h[0])
        elif d == 2:
            from scipy.interpolate import RegularGridInterpolator
            interp = RegularGridInterpolator(grids_ref, phi_ref, method='cubic')
            pts = np.array(list(product(grids_h[0], grids_h[1])))
            phi_ref_on_coarse = interp(pts).reshape(phi_h.shape)
        elif d == 3:
            from scipy.interpolate import RegularGridInterpolator
            interp = RegularGridInterpolator(grids_ref, phi_ref, method='linear')
            pts = np.array(list(product(grids_h[0], grids_h[1], grids_h[2])))
            phi_ref_on_coarse = interp(pts).reshape(phi_h.shape)
        
        err = phi_h - phi_ref_on_coarse
        h = np.pi / (N + 1)
        e_inf = np.max(np.abs(err))
        e_L2 = np.sqrt(h**d * np.sum(err**2))
        
        results.append({'N': N, 'h': h, 'e_inf': e_inf, 'e_L2': e_L2})
    
    # Compute rates
    for i in range(1, len(results)):
        r = results[i]
        r_prev = results[i-1]
        r['rate_inf'] = np.log2(r_prev['e_inf'] / r['e_inf'])
        r['rate_L2'] = np.log2(r_prev['e_L2'] / r['e_L2'])
    results[0]['rate_inf'] = None
    results[0]['rate_L2'] = None
    
    return results


def print_table(results, d):
    """Print a LaTeX-formatted convergence table."""
    print(f"\n=== d = {d} ===")
    print(f"{'N':>6} {'e_inf':>12} {'Rate':>8} {'e_L2':>12} {'Rate':>8}")
    print("-" * 50)
    for r in results:
        rate_inf = f"{r['rate_inf']:.2f}" if r['rate_inf'] is not None else "---"
        rate_L2 = f"{r['rate_L2']:.2f}" if r['rate_L2'] is not None else "---"
        print(f"{r['N']:>6} {r['e_inf']:>12.4e} {rate_inf:>8} {r['e_L2']:>12.4e} {rate_L2:>8}")
    
    # Also print LaTeX
    print(f"\n% LaTeX table for d={d}:")
    for r in results:
        rate_inf = f"{r['rate_inf']:.2f}" if r['rate_inf'] is not None else "---"
        rate_L2 = f"{r['rate_L2']:.2f}" if r['rate_L2'] is not None else "---"
        print(f"{r['N']} & ${r['e_inf']:.2e}$ & {rate_inf} & ${r['e_L2']:.2e}$ & {rate_L2} \\\\")


def energy(phi_curr, phi_prev, dt, h, c, lam, d):
    """Compute discrete energy."""
    # Kinetic: (1/2) * sum (Φ^n - Φ^{n-1})² / dt²
    KE = 0.5 * h**d * np.sum(((phi_curr - phi_prev) / dt)**2)
    
    # Gradient: (c²/2) * sum |∇_h Φ|²
    grad_sq = np.zeros_like(phi_curr)
    shape_padded = tuple(s + 2 for s in phi_curr.shape)
    u_pad = np.zeros(shape_padded)
    interior = tuple(slice(1, -1) for _ in range(d))
    u_pad[interior] = phi_curr
    for axis in range(d):
        slc = [slice(1, -1)] * d
        slc[axis] = slice(2, None)
        grad_sq += ((u_pad[tuple(slc)] - phi_curr) / h)**2
    GE = 0.5 * c**2 * h**d * np.sum(grad_sq)
    
    # Nonlinear: (λ/4) * sum Φ⁴
    NE = 0.25 * lam * h**d * np.sum(phi_curr**4)
    
    return KE + GE + NE


if __name__ == "__main__":
    print("=" * 60)
    print("FINITE DIFFERENCE CONVERGENCE TESTS")
    print("Equation: Φ_tt = c² ΔΦ - λ Φ³")
    print("Domain: [0,π]^d, Dirichlet BCs")
    print("Scheme: Leapfrog, CFL = 0.9/√d")
    print("=" * 60)
    
    # d = 1
    print("\n--- d = 1 ---")
    r1 = convergence_test(1, [16, 32, 64, 128, 256], N_ref=512)
    print_table(r1, 1)
    
    # d = 2
    print("\n--- d = 2 ---")
    r2 = convergence_test(2, [8, 16, 32, 64], N_ref=128)
    print_table(r2, 2)
    
    # d = 3
    print("\n--- d = 3 ---")
    r3 = convergence_test(3, [8, 16, 32], N_ref=64)
    print_table(r3, 3)
    
    print("\nDone. Copy the LaTeX tables into Appendix E.")
```

### 1.5 What to Check

Before trusting results, verify these sanity checks:

1. **Linear test first.** Set `λ = 0`. The exact solution is `Φ(x,t) = ε sin(x) cos(t)` (in 1D with `c=1`). The error should be exactly `O(h² + dt²)` with known constant. If you don't get rate 2.00 for the linear case, the code has a bug.

2. **Energy conservation.** For the nonlinear case, compute the discrete energy at each time step. It should drift by `O(dt)` per unit time (not blow up). Plot `|E(t) - E(0)|` vs `t` to verify.

3. **CFL violation test.** Set `ν = 1.1/√d` (above the CFL limit). The solution should blow up. If it doesn't, the Laplacian implementation is wrong.

4. **Symmetry.** The solution should respect the symmetry of the initial data. In 2D, `Φ(x,y) = Φ(y,x)` should hold to machine precision.

### 1.6 Expected Runtime

- d=1, N=512: ~1 second
- d=2, N=128: ~30 seconds  
- d=3, N=64: ~5 minutes (64³ = 262,144 grid points × ~3000 time steps)

Total for all convergence tests: under 15 minutes on a modern laptop.

### 1.7 Output

Produce three LaTeX tables (one per dimension) in the exact format of the commented-out tables in Appendix E. Each table has columns: N, `‖e‖_∞`, Rate, `‖e‖_{L²}`, Rate. Also produce one energy conservation plot showing `|E(t) - E(0)|/E(0)` vs `t` for the 1D case at N=128.

---

## Module 2: Spectral Method Convergence (Appendix F)

### 2.1 The Method

Galerkin projection onto Chebyshev polynomials `T_n` on `[-1, 1]` (rescaled to `[0, π]`). The solution is expanded as:

```
Φ_N(x, t) = Σ_{n=0}^{N} c_n(t) φ_n(x)
```

where `φ_n(x) = sin(n π x / π) = sin(n x)` (sine series for Dirichlet BCs, equivalent to Chebyshev of the second kind on this domain).

The Galerkin equations are the ODE system:

```
c̈_n(t) = -c² n² c_n(t) + F̂_n({c_m})
```

where `F̂_n` is the projection of `-λ Φ_N³ + V Φ_N` onto mode `n`. The cubic term couples modes via the convolution sum.

### 2.2 Implementation Approach

For the sine basis on `[0, π]`, the Galerkin projection of `Φ³` can be computed via the discrete sine transform (DST):

```python
import numpy as np
from scipy.fft import dstn, idstn

def spectral_rhs(c_coeffs, N, c_wave, lam):
    """
    Compute RHS of the Galerkin ODE system.
    c_coeffs: array of N spectral coefficients
    Returns: array of N accelerations c̈_n
    """
    # Reconstruct Φ on a fine grid (for the nonlinear term)
    N_grid = 3 * N  # 3/2 rule for dealiasing
    phi_grid = idstn(c_coeffs, type=1)  # inverse DST to get physical values
    
    # Nonlinear term in physical space
    F_grid = -lam * phi_grid**3
    
    # Project back to spectral space
    F_coeffs = dstn(F_grid, type=1)  # forward DST
    
    # Linear term: -c² n² c_n (exact in spectral space)
    n_arr = np.arange(1, N + 1)
    linear = -c_wave**2 * n_arr**2 * c_coeffs
    
    return linear + F_coeffs[:N]
```

Time-step the ODE system with RK4 or leapfrog (for energy comparisons).

### 2.3 Tests to Run

**Linear case (λ = 0), d = 1:**
- N = 4, 8, 12, 16, 20, 24, 32
- Compare against exact solution `Φ(x,t) = ε sin(x) cos(t)`
- Plot `log(error)` vs `N`: should be a straight line (exponential convergence)
- Extract the rate `α` from the slope

**Nonlinear case (λ = 1), d = 1:**
- N = 4, 8, 12, 16, 20, 24, 32
- Compare against N=64 reference
- Plot `log(error)` vs `N`: should still be approximately linear (exponential)
- Verify the rate `α(T)` decreases with T (run at T=1, T=5, T=10)

**Cost comparison:**
- For each target accuracy `ε`, record the smallest N (spectral) and smallest h (FD) that achieves `‖e‖ < ε`
- Tabulate for `ε = 10⁻⁴, 10⁻⁶, 10⁻⁸`
- This fills the comparison in Table 8 of the main paper

### 2.4 Expected Output

One semilog plot (error vs N) showing exponential convergence for both linear and nonlinear cases. One table comparing FD and spectral costs.

---

## Module 3: Physical Parameter Estimates (Appendix G)

### 3.1 What to Compute

For each beam type (electron at 10, 50, 100 MeV; proton at 100 MeV, 1 GeV), compute:

```python
import numpy as np

# Constants (SI)
m_e = 9.109e-31     # electron mass (kg)
m_p = 1.673e-27     # proton mass (kg)
e = 1.602e-19        # elementary charge (C)
eps0 = 8.854e-12     # vacuum permittivity (F/m)
c = 2.998e8          # speed of light (m/s)
hbar = 1.055e-34     # reduced Planck constant (J·s)

def beam_parameters(m, E_MeV, n_cm3, sigma_v_over_c):
    """
    Compute collective mode parameters for a beam.
    
    Parameters
    ----------
    m : float, particle mass in kg
    E_MeV : float, beam energy in MeV
    n_cm3 : float, beam density in particles/cm³
    sigma_v_over_c : float, velocity spread as fraction of c
    """
    E_J = E_MeV * 1.602e-13  # convert MeV to Joules
    gamma = E_J / (m * c**2) + 1  # Lorentz factor (total energy / rest energy)
    v0 = c * np.sqrt(1 - 1/gamma**2)  # beam velocity
    m_eff = gamma * m  # effective mass
    n_SI = n_cm3 * 1e6  # convert to m⁻³
    sigma_v = sigma_v_over_c * c
    
    # Plasma frequency
    Omega_p = np.sqrt(n_SI * e**2 / (m_eff * eps0))
    f_p = Omega_p / (2 * np.pi)  # in Hz
    
    # Critical density (for Gaussian beam)
    # n_c = m_eff * eps0 * sigma_v² * q² / e²
    # At the characteristic q ~ Omega_p / v0:
    q_char = Omega_p / v0
    n_c = m_eff * eps0 * sigma_v**2 * q_char**2 / e**2
    n_c_cm3 = n_c / 1e6
    
    # Compton wavelength
    lambda_C = hbar / (m * c)
    
    # Maximum wavevector
    q_max = Omega_p / sigma_v
    lambda_min = 2 * np.pi / q_max  # minimum wavelength
    
    # Damping rate (Gaussian beam, exponentially small)
    # Gamma ~ exp(-Omega_p² / (2 sigma_v² q²)) at q ~ q_max
    log_Gamma = -Omega_p**2 / (2 * sigma_v**2 * q_max**2)
    
    # Correlation length at 10% off criticality
    # xi ~ 1/sqrt(n - n_c) * (some scale)
    
    return {
        'gamma': gamma,
        'v0/c': v0/c,
        'm_eff/m': gamma,
        'Omega_p_Hz': f_p,
        'Omega_p_eV': hbar * Omega_p / e,
        'lambda_C_m': lambda_C,
        'q_max_m-1': q_max,
        'lambda_min_m': lambda_min,
        'n_c_cm-3': n_c_cm3,
        'log10_Gamma/Omega_p': log_Gamma / np.log(10),
    }

# Example: 50 MeV electron beam, n = 10^12 cm^-3, sigma_v = 0.01c
params = beam_parameters(m_e, 50, 1e12, 0.01)
for k, v in params.items():
    print(f"  {k}: {v:.4g}")
```

### 3.2 Table to Produce

Run the above for this parameter grid:

| Particle | E (MeV) | n (cm⁻³) | σ_v/c |
|----------|---------|----------|-------|
| e⁻ | 10 | 10¹⁰, 10¹¹, 10¹² | 0.01 |
| e⁻ | 50 | 10¹⁰, 10¹¹, 10¹² | 0.01 |
| e⁻ | 100 | 10¹⁰, 10¹¹, 10¹² | 0.01 |
| p | 100 | 10¹⁰, 10¹¹, 10¹² | 0.01 |
| p | 1000 | 10¹⁰, 10¹¹, 10¹² | 0.01 |

For each row, report: γ, Ω_p (eV), n_c (cm⁻³), q_max (m⁻¹), λ_min (μm), and whether n > n_c (i.e., whether collective modes are expected).

---

## Execution Order

The three modules are independent. Suggested order by priority:

1. **Module 1** (FD convergence) — this is the one the paper explicitly promises and a referee will check. Start here. The code skeleton above is nearly complete; you mainly need to debug the Laplacian, run it, and paste the tables into Appendix E.

2. **Module 3** (parameter estimates) — fast to run (just arithmetic), and it fills Section 5 (experimental predictions) with concrete numbers. This is what makes the paper physically grounded rather than purely mathematical.

3. **Module 2** (spectral convergence) — nice to have but lower priority. The theoretical argument in the paper is self-contained; the numerics just confirm it. A referee is unlikely to insist on this if Module 1 is done.

---

## File Organization

```
validation/
├── fd_convergence.py          # Module 1: finite difference tests
├── spectral_convergence.py    # Module 2: spectral method tests  
├── parameter_estimates.py     # Module 3: physical parameters
├── results/
│   ├── conv_1d.tex            # LaTeX table for d=1
│   ├── conv_2d.tex            # LaTeX table for d=2
│   ├── conv_3d.tex            # LaTeX table for d=3
│   ├── energy_drift.pdf       # Energy conservation plot
│   ├── spectral_conv.pdf      # Spectral convergence plot
│   └── parameters.tex         # Physical parameter table
└── README.md                  # This file
```

---

## Acceptance Criteria

The numerical results are ready for the paper when:

1. All convergence rates are within 0.05 of 2.00 (for FD) or show a clean straight line on a semilog plot (for spectral). If they deviate, the code has a bug — do not adjust the theory to match wrong numerics.

2. The energy drift test shows `|ΔE/E| < 10⁻⁴` over T = 1 for the finest grid. If not, the time step is too large or the Laplacian is wrong.

3. The CFL violation test produces visible blowup within 100 time steps.

4. The linear test (λ = 0) recovers the exact analytical solution to machine precision (modulo O(h²) discretization error).

5. The parameter estimates produce n_c values in the range 10⁹ - 10¹² cm⁻³ for electron beams at 10-100 MeV (if they give n_c = 10²⁰ or n_c = 1, something is wrong with the formula or units).
