"""DESC fixed-boundary solve started from a VMEC++ solution: python 12_desc_from_vmec.py L M N WOUT [maxiter]
The wout is read with DESC's VMECIO at (L, M, N); its profiles are replaced by the exact c08 power series (pressure,
current) of reactor_QA_c08.h5. The DESC force error is reported for the start (the VMEC++ solution as DESC sees it)
and for the solution, which is saved as desc_fromvmec_L*_M*_N*.h5/.pkl."""
import os, sys, time, pickle
import numpy as np
import desc.io
from desc.vmec import VMECIO
from desc.grid import ConcentricGrid
from desc.objectives import ForceBalance, ObjectiveFunction, get_fixed_boundary_constraints
from common import desc_eval

L, M, N, wout = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
maxiter = int(sys.argv[5]) if len(sys.argv) > 5 else 50
ref = desc.io.load("reactor_QA_c08.h5")
eq = VMECIO.load(wout, L=L, M=M, N=N, profile="current")
eq.pressure, eq.current, eq.iota = ref.pressure, ref.current, None
eq.change_resolution(L_grid=2 * L, M_grid=2 * M, N_grid=2 * N)
eq.Psi = ref.Psi
f = lambda e: e.compute(["<|F|>_vol", "<|grad(|B|^2)|/2mu0>_vol"])
d = f(eq); print("start (VMEC++ solution in DESC): normalized <|F|> = %.3e" % (d["<|F|>_vol"] / d["<|grad(|B|^2)|/2mu0>_vol"]), flush=True)
pickle.dump(desc_eval(eq), open("desc_vmecstart_L%d_M%d_N%d.pkl" % (L, M, N), "wb"))
grid = ConcentricGrid(eq.L_grid, eq.M_grid, eq.N_grid, NFP=eq.NFP, sym=eq.sym, node_pattern="jacobi")
obj = ObjectiveFunction(ForceBalance(eq=eq, grid=grid))
t = time.time()
eq, res = eq.solve(objective=obj, constraints=get_fixed_boundary_constraints(eq=eq), verbose=3, ftol=1e-8, xtol=1e-10,
                   gtol=1e-10, maxiter=maxiter, copy=True)
tag = "desc_fromvmec_L%d_M%d_N%d" % (L, M, N)
print("%s: solve %.0fs nfev %d" % (tag, time.time() - t, res["nfev"]), flush=True)
eq.save(tag + ".h5"); r = desc_eval(eq); pickle.dump(r, open(tag + ".pkl", "wb"))
print("solution: normalized <|F|> = %.3e" % r["force"])
