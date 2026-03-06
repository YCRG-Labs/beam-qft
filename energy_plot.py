import numpy as np
import matplotlib.pyplot as plt

from fd_solver import leapfrog_solver


def energy_drift_test():

    N = 256
    T = 1.0

    x, phi, energies, dt = leapfrog_solver(
        N,
        T,
        store_energy=True
    )

    E0 = energies[0]

    drift = np.abs((energies - E0) / E0)

    print("Final energy drift:", drift[-1])

    t = np.arange(len(energies))*dt

    plt.plot(t, drift)

    plt.xlabel("time")
    plt.ylabel("|ΔE/E|")

    plt.title("Energy Drift")

    plt.show()


if __name__ == "__main__":

    energy_drift_test()