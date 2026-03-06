import numpy as np
from itertools import product

def leapfrog_solve(d, N, c, lam, V_func, phi0_func, T, cfl_frac=0.9):
    """
    Solve ∂²Φ/∂t² = c² ΔΦ - λ Φ³ + V(x) Φ on [0,π]^d
    with Dirichlet BCs using leapfrog (Störmer-Verlet).
    """
    h = np.pi / (N + 1)
    dt = cfl_frac * h / (c * np.sqrt(d))
    Nt = int(np.ceil(T / dt))
    dt = T / Nt  # adjust to hit T exactly

    # Grid coordinates
    x1d = np.linspace(h, np.pi - h, N)
    if d == 1:
        X = x1d
        phi = phi0_func(X)
    elif d == 2:
        X, Y = np.meshgrid(x1d, x1d, indexing='ij')
        phi = phi0_func(X, Y)
    else:
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
        """Zero-padded discrete Laplacian (2nd order, Dirichlet)."""
        shape_pad = tuple(s + 2 for s in u.shape)
        u_pad = np.zeros(shape_pad)
        interior = tuple(slice(1, -1) for _ in range(d))
        u_pad[interior] = u

        lap = np.zeros_like(u)
        for axis in range(d):
            slc_plus = list(interior)
            slc_minus = list(interior)
            slc_plus[axis] = slice(2, None)
            slc_minus[axis] = slice(0, -2)
            lap += (u_pad[tuple(slc_plus)] + u_pad[tuple(slc_minus)] - 2 * u) / h**2
        return lap

    def rhs(u):
        return c**2 * laplacian(u) - lam * u**3 + V * u

    # First step (Taylor expansion)
    phi_prev = phi.copy()
    phi_curr = phi + 0.5 * dt**2 * rhs(phi)

    # Leapfrog iterations
    energy_list = []
    for n in range(1, Nt):
        phi_next = 2 * phi_curr - phi_prev + dt**2 * rhs(phi_curr)
        phi_prev, phi_curr = phi_curr, phi_next
        energy_list.append(energy(phi_curr, phi_prev, dt, h, c, lam, d))

    if d == 1:
        return phi_curr, (x1d,), np.array(energy_list)
    elif d == 2:
        return phi_curr, (x1d, x1d), np.array(energy_list)
    else:
        return phi_curr, (x1d, x1d, x1d), np.array(energy_list)


def energy(phi_curr, phi_prev, dt, h, c, lam, d):
    """Discrete energy: kinetic + gradient + nonlinear."""
    KE = 0.5 * h**d * np.sum(((phi_curr - phi_prev)/dt)**2)

    # Gradient energy using zero-padding
    shape_pad = tuple(s + 2 for s in phi_curr.shape)
    u_pad = np.zeros(shape_pad)
    interior = tuple(slice(1, -1) for _ in range(d))
    u_pad[interior] = phi_curr

    grad_sq = np.zeros_like(phi_curr)
    for axis in range(d):
        slc_plus = list(interior)
        slc_plus[axis] = slice(2, None)
        grad_sq += ((u_pad[tuple(slc_plus)] - phi_curr)/h)**2
    GE = 0.5 * c**2 * h**d * np.sum(grad_sq)

    NE = 0.25 * lam * h**d * np.sum(phi_curr**4)
    return KE + GE + NE