"""Step 1: reactor_QA with a current profile that is even in rho (analytic in s), as VMEC requires.

Modification of desc/examples/reactor_QA.py: the example uses
    PowerSeriesProfile(np.zeros(L+1), np.arange(L+1), sym=False)   -> I ~ c2 rho^2 + c3 rho^3 + ...
The odd powers of rho are not analytic at the axis (I must be a power series in s = rho^2) and cannot be
represented by VMEC's power_series current. Here the current has only even powers, rho^2 ... rho^(2L),
with c0 = 0 fixed, and the bootstrap-consistency optimisation of the example is repeated (warm-started from the
shipped example output). The kinetic profiles are also collapsed to the equivalent pressure polynomial
p = e n (Te + Ti), which is exact (n ~ 1 - rho^10, T ~ 1 - rho^2).
"""
import sys, time
import numpy as np
from scipy.constants import elementary_charge as qe
from desc.examples import get
from desc.grid import LinearGrid
from desc.objectives import (BootstrapRedlConsistency, FixAtomicNumber, FixBoundaryR, FixBoundaryZ, FixCurrent,
                             FixElectronDensity, FixElectronTemperature, FixIonTemperature, FixPsi, ForceBalance,
                             ObjectiveFunction)
from desc.profiles import PowerSeriesProfile

maxiter = int(sys.argv[1]) if len(sys.argv) > 1 else 10
eq = get("reactor_QA")
rho = np.linspace(0, 1, 201)
I_orig = eq.current(rho)

# --- even current: least-squares fit of the example's profile as the starting point
modes = np.arange(0, 2 * eq.L + 1, 2)
A = rho[:, None] ** modes[None, 1:]
c = np.linalg.lstsq(A, I_orig, rcond=None)[0]
eq.current = PowerSeriesProfile(np.concatenate([[0.0], c]), modes, sym="even")
print("fit: max |I_even - I_orig| / max|I| = %.3e" % (np.abs(eq.current(rho) - I_orig).max() / np.abs(I_orig).max()))

t = time.time()
eq, _ = eq.solve(verbose=2, copy=True, ftol=1e-10, xtol=1e-12, gtol=1e-12, maxiter=100)
print("fixed-boundary re-solve %.0fs" % (time.time() - t), flush=True)

# --- the example's bootstrap optimisation, with the even current
grid = LinearGrid(rho=np.linspace(1 / eq.L_grid, 1, eq.L_grid) - 1 / (2 * eq.L_grid), M=eq.M_grid, N=eq.N_grid,
                  NFP=eq.NFP, sym=True)
objective = ObjectiveFunction(BootstrapRedlConsistency(eq=eq, grid=grid, helicity=(1, 0)))
constraints = (FixAtomicNumber(eq=eq), FixBoundaryR(eq=eq), FixBoundaryZ(eq=eq),
               FixCurrent(eq=eq, indices=np.array([0])),      # I(0) = 0; odd powers do not exist
               FixElectronDensity(eq=eq), FixElectronTemperature(eq=eq), FixIonTemperature(eq=eq), FixPsi(eq=eq),
               ForceBalance(eq=eq))
t = time.time()
eq, _ = eq.optimize(objective=objective, constraints=constraints, optimizer="proximal-lsq-exact", maxiter=maxiter,
                    gtol=1e-16, verbose=3, copy=True)
print("bootstrap optimisation %.0fs" % (time.time() - t), flush=True)
print("current modes", eq.current.basis.modes[:, 0], "\ncoefficients", eq.current.params)

# --- kinetic profiles -> pressure polynomial (exact), then a final tight fixed-boundary solve
ne, Te, Ti = eq.electron_density, eq.electron_temperature, eq.ion_temperature
p_kin = eq.compute("p", grid=LinearGrid(rho=rho, M=0, N=0))["p"]
p0 = qe * ne.params[0] * (Te.params[0] + Ti.params[0])
pm = np.zeros(13); pm[[0, 2, 10, 12]] = p0 * np.array([1, -1, -1, 1])   # (1 - rho^10)(1 - rho^2)
cur = eq.current
eq.electron_density = None; eq.electron_temperature = None; eq.ion_temperature = None; eq.atomic_number = None
eq.pressure = PowerSeriesProfile(pm[::2], np.arange(0, 13, 2), sym="even")
eq.current = cur
print("pressure: max rel diff to kinetic %.2e" % (np.abs(eq.pressure(rho) - p_kin).max() / p0))
eq, _ = eq.solve(verbose=2, copy=True, ftol=1e-10, xtol=1e-12, gtol=1e-12, maxiter=100)
d = eq.compute(["<beta>_vol", "a", "R0", "<|B|>_vol", "<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"])
print({k: float(d[k]) for k in ["<beta>_vol", "a", "R0", "<|B|>_vol"]},
      "normalised force %.3e" % (d["<|F|>_vol"] / d["<|grad(|B|^2)|/2mu0>_vol"]))
eq.save("reactor_QA_even.h5")
np.savez("current_profiles.npz", rho=rho, I_orig=I_orig, I_even=eq.current(rho))
