"""DESC free-boundary run: python 04_desc_free.py L M N START_FIXED.h5 [maxiter]

Starts from the fixed-boundary solution START_FIXED.h5 at the same resolution and solves force balance with the
boundary free, B.n = 0 and pressure balance on the LCFS against the external field of coil_field.h5 (BoundaryError,
virtual casing for the plasma field), with the proximal-lsq-exact optimizer as in DESC's free-boundary tutorial.
"""
import os, sys, time, pickle
import numpy as np
import desc, desc.io
# cap DESC's memory estimate so 'auto' Jacobian chunking of the inner equilibrium constraint fits (default: all CPU memory)
if os.environ.get("DESC_MEM"):
    desc.config["avail_mem"] = float(os.environ["DESC_MEM"])
from desc.grid import LinearGrid
from desc.objectives import BoundaryError, FixCurrent, FixPressure, FixPsi, ForceBalance, ObjectiveFunction
from common import load_field, desc_eval

L, M, N, start = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
maxiter = int(sys.argv[5]) if len(sys.argv) > 5 else 30
tag = "desc_free_L%d_M%d_N%d" % (L, M, N)
field, src = load_field()
eq = desc.io.load(start)
assert (eq.L, eq.M, eq.N) == (L, M, N), (eq.L, eq.M, eq.N)
t0 = time.time()
GM = int(os.environ.get("BGRID", 2 * M))
bgrid = LinearGrid(M=GM, N=GM, NFP=eq.NFP)
print("boundary-error grid M = N = %d" % GM, flush=True)
objective = ObjectiveFunction(BoundaryError(eq=eq, field=field, field_grid=src, field_fixed=True, source_grid=bgrid,
                                            eval_grid=bgrid, bs_chunk_size=int(os.environ.get("BS_CHUNK", 64)),
                                            B_plasma_chunk_size=int(os.environ.get("BP_CHUNK", 256))),
                              jac_chunk_size=int(os.environ.get("JAC_CHUNK", 32)))
constraints = (ForceBalance(eq=eq), FixCurrent(eq=eq), FixPressure(eq=eq), FixPsi(eq=eq))
eq, res = eq.optimize(objective, constraints, optimizer="proximal-lsq-exact", maxiter=maxiter,
                      ftol=float(os.environ.get("OUTER_FTOL", 1e-6)), xtol=1e-10, gtol=1e-16, verbose=3, copy=True,
                      options={"solve_options": {"ftol": float(os.environ.get("INNER_FTOL", 1e-8)), "xtol": 1e-10, "gtol": 1e-10,
                                                 "maxiter": int(os.environ.get("INNER_MAXITER", 50))}})
print("%s: free-boundary optimisation %.0fs" % (tag, time.time() - t0), flush=True)
eq.save(tag + "_unpolished.h5")
# polish: tight fixed-boundary solve on the free-boundary LCFS
t1 = time.time()
eq, _ = eq.solve(verbose=1, ftol=1e-8, xtol=1e-10, gtol=1e-10, maxiter=int(os.environ.get("POLISH_MAXITER", 50)), copy=True)
print("%s: polish solve %.0fs" % (tag, time.time() - t1), flush=True)
eq.save(tag + ".h5")
r = desc_eval(eq); r["wall"] = time.time() - t0
pickle.dump(r, open(tag + ".pkl", "wb"))
print({k: v for k, v in r.items() if np.isscalar(v)}, "iota", r["iota"][[0, -1]])
