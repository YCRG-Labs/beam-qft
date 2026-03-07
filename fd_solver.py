import numpy as np


def leapfrog_solve(d, N, c, lam, V_func, phi0_func, T, cfl_frac=0.9, store_energy=False):
    """
    Solve  dtt Phi = c^2 Lap Phi - lam Phi^3 + V(x) Phi  on [0,pi]^d
    with Dirichlet BCs using leapfrog (Stormer-Verlet).

    Parameters
    ----------
    d : int
        Spatial dimension (1, 2, or 3).
    N : int
        Number of interior grid points per dimension.
    c : float
        Wave speed.
    lam : float
        Nonlinear coupling (lam > 0 for defocusing).
    V_func : callable or None
        External potential V(x).
    phi0_func : callable
        Initial data Phi_0(x).
    T : float
        Final time.
    cfl_frac : float
        Fraction of CFL limit to use (default 0.9).
    store_energy : bool
        If True, compute and return energy at each time step.

    Returns
    -------
    phi_curr : ndarray, shape (N,)*d
        Solution at time T on interior grid points.
    grids : tuple of 1d arrays
        Grid coordinates in each dimension.
    energies : ndarray or None
        Energy time series if store_energy=True, else None.
    dt : float
        Time step used.
    """
    h = np.pi / (N + 1)
    dt = cfl_frac * h / (c * np.sqrt(d))
    Nt = int(np.ceil(T / dt))
    dt = T / Nt

    # Grid coordinates (interior points only)
    x1d = np.linspace(h, np.pi - h, N)
    if d == 1:
        coords = (x1d,)
        phi = phi0_func(x1d)
    elif d == 2:
        X, Y = np.meshgrid(x1d, x1d, indexing='ij')
        coords = (x1d, x1d)
        phi = phi0_func(X, Y)
    else:
        X, Y, Z = np.meshgrid(x1d, x1d, x1d, indexing='ij')
        coords = (x1d, x1d, x1d)
        phi = phi0_func(X, Y, Z)

    # Precompute V on grid
    if V_func is not None:
        if d == 1:
            V = V_func(x1d)
        elif d == 2:
            V = V_func(X, Y)
        else:
            V = V_func(X, Y, Z)
    else:
        V = 0.0

    def laplacian(u):
        """Zero-padded discrete Laplacian (2nd order, Dirichlet BCs)."""
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

    def potential_energy(u):
        """Potential energy U(q) = (c^2/2)|grad q|^2 + (lam/4) q^4."""
        shape_pad = tuple(s + 2 for s in u.shape)
        u_pad = np.zeros(shape_pad)
        interior = tuple(slice(1, -1) for _ in range(d))
        u_pad[interior] = u

        grad_sq = np.zeros_like(u)
        for axis in range(d):
            slc_plus = list(interior)
            slc_plus[axis] = slice(2, None)
            grad_sq += ((u_pad[tuple(slc_plus)] - u) / h) ** 2
        GE = 0.5 * c**2 * h**d * np.sum(grad_sq)
        NE = 0.25 * lam * h**d * np.sum(u**4)
        return GE + NE

    def energy_shadow(phi_next, phi_curr):
        """
        Shadow Hamiltonian at half-step n+1/2.

        For Stormer-Verlet, the exactly conserved quantity (up to O(dt^2)
        bounded oscillations) pairs the forward half-step velocity
        v^{n+1/2} = (q^{n+1} - q^n) / dt with the potential at q^n.

        This has NO secular drift. The centered velocity (q^{n+1} - q^{n-1})/(2dt)
        and the backward velocity (q^n - q^{n-1})/dt paired with U(q^n) both
        introduce O(dt) mismatches that accumulate.
        """
        v = (phi_next - phi_curr) / dt
        KE = 0.5 * h**d * np.sum(v**2)
        PE = potential_energy(phi_curr)
        return KE + PE

    # First step (Taylor expansion, since dphi/dt(0) = 0)
    phi_prev = phi.copy()
    phi_curr = phi + 0.5 * dt**2 * rhs(phi)

    energy_list = []

    # Leapfrog iterations
    for n in range(1, Nt):
        phi_next = 2 * phi_curr - phi_prev + dt**2 * rhs(phi_curr)

        if store_energy:
            # Shadow Hamiltonian at half-step n+1/2:
            # KE from v^{n+1/2} = (phi_next - phi_curr)/dt
            # PE from phi_curr = q^n
            energy_list.append(energy_shadow(phi_next, phi_curr))

        phi_prev = phi_curr
        phi_curr = phi_next

    energies = np.array(energy_list) if store_energy else None
    return phi_curr, coords, energies, dt
