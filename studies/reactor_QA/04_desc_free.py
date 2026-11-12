"""DESC free-boundary run of reactor_QA_even: python 04_desc_free.py L M N [start.h5] [maxiter]"""
import sys, time, pickle, os
import numpy as np
import desc.io
from desc.grid import LinearGrid
from desc.objectives import BoundaryError, FixCurrent, FixPressure, FixPsi, ForceBalance, ObjectiveFunction
from common import load_field, desc_eval

L, M, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
start = sys.argv[4] if len(sys.argv) > 4 else "reactor_QA_even.h5"
maxiter = int(sys.argv[5]) if len(sys.argv) > 5 else 30
tag = "desc_free_L%d_M%d_N%d" % (L, M, N)
field, src = load_field()
eq = desc.io.load(start)
eq.change_resolution(L=L, M=M, N=N, L_grid=2 * L, M_grid=2 * M, N_grid=2 * N)
t0 = time.time()
if (L, M, N) != (8, 8, 8) or start == "reactor_QA_even.h5":
    eq, _ = eq.solve(verbose=1, ftol=1e-10, xtol=1e-12, gtol=1e-12, maxiter=150, copy=True)   # fixed-boundary at this resolution
    eq.save("desc_fixed_L%d_M%d_N%d.h5" % (L, M, N))
    pickle.dump(desc_eval(eq), open("desc_fixed_L%d_M%d_N%d.pkl" % (L, M, N), "wb"))
    print("fixed-boundary solve %.0fs" % (time.time() - t0), flush=True)

bgrid = LinearGrid(M=2 * M, N=2 * N, NFP=eq.NFP)
objective = ObjectiveFunction(BoundaryError(eq=eq, field=field, field_grid=src, field_fixed=True, source_grid=bgrid,
                                            eval_grid=bgrid, bs_chunk_size=64, B_plasma_chunk_size=256),
                              jac_chunk_size=32)
constraints = (ForceBalance(eq=eq), FixCurrent(eq=eq), FixPressure(eq=eq), FixPsi(eq=eq))
t1 = time.time()
eq, res = eq.optimize(objective, constraints, optimizer="proximal-lsq-exact", maxiter=maxiter, ftol=1e-6, xtol=1e-10,
                      gtol=1e-16, verbose=3, copy=True,
                      options={"solve_options": {"ftol": 1e-10, "xtol": 1e-12, "gtol": 1e-12, "maxiter": 100}})
print("free-boundary optimisation %.0fs" % (time.time() - t1), flush=True)
eq.save(tag + ".h5")
r = desc_eval(eq); r["wall"] = time.time() - t0
pickle.dump(r, open(tag + ".pkl", "wb"))
print({k: v for k, v in r.items() if np.isscalar(v)}, "iota", r["iota"][[0, -1]])
