"""Fixed-boundary DESC solve of reactor_QA_even: python 05_desc_fixed.py L M N {jacobi|jacobi-dna}

The ForceBalance grid is ConcentricGrid(L_grid, M_grid, N_grid) with the given radial node pattern.
'jacobi-dna' is the node pattern of the DESC branch dp/iota-calc-in-force-bal (patched into DESC 0.17.3): the
innermost rho ring has as many nodes as the next ring, so the flux-surface averages that give iota from the
enclosed current are accurate near the axis.
"""
import os, sys, time, pickle
import numpy as np
import desc.io
from desc.grid import ConcentricGrid
from desc.objectives import ForceBalance, ObjectiveFunction, get_fixed_boundary_constraints
from common import desc_eval

L, M, N, pattern = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
tag = "desc_fixed_L%d_M%d_N%d_%s" % (L, M, N, pattern)
eq = desc.io.load(sys.argv[5] if len(sys.argv) > 5 else "reactor_QA_even.h5")
eq.change_resolution(L=L, M=M, N=N, L_grid=2 * L, M_grid=2 * M, N_grid=2 * N)
grid = ConcentricGrid(eq.L_grid, eq.M_grid, eq.N_grid, NFP=eq.NFP, sym=eq.sym, node_pattern=pattern)
print("grid nodes", grid.num_nodes, "innermost rho %.4f with %d nodes" % (grid.nodes[0, 0], np.sum(grid.nodes[:, 0] == grid.nodes[0, 0])))
obj = ObjectiveFunction(ForceBalance(eq=eq, grid=grid), jac_chunk_size=int(sys.argv[6]) if len(sys.argv) > 6 else "auto")
t = time.time()
eq, res = eq.solve(objective=obj, constraints=get_fixed_boundary_constraints(eq=eq), verbose=3, ftol=float(os.environ.get("FTOL", 1e-8)),
                   xtol=float(os.environ.get("XTOL", 1e-10)), gtol=1e-10, maxiter=int(os.environ.get("MAXITER", 100)), copy=True)
print("%s: solve %.0fs, nfev %d" % (tag, time.time() - t, res["nfev"]), flush=True)
eq.save(tag + ".h5")
r = desc_eval(eq); r["wall"] = time.time() - t
pickle.dump(r, open(tag + ".pkl", "wb"))
print({k: v for k, v in r.items() if np.isscalar(v)}, "iota", r["iota"][[0, -1]])
