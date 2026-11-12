"""Quadrature convergence of DESC's BoundaryError (virtual casing + coil field) in the boundary grid M=N=GM."""
import os, sys, time
import numpy as np
import desc, desc.io
desc.config["avail_mem"] = 6.0
from desc.grid import LinearGrid
from desc.objectives import BoundaryError
from common import load_field
field, src = load_field()
for f in sys.argv[1:]:
    eq = desc.io.load(f)
    for GM in [24, 32, 40, 48]:
        t = time.time()
        g = LinearGrid(M=GM, N=GM, NFP=eq.NFP)
        obj = BoundaryError(eq=eq, field=field, field_grid=src, field_fixed=True, source_grid=g, eval_grid=g, bs_chunk_size=64, B_plasma_chunk_size=64)
        obj.build(verbose=0)
        r = np.asarray(obj.compute_scaled_error(*obj.xs(eq)))
        n = r.size // 2
        fu = np.asarray(obj.compute_unscaled(*obj.xs(eq)))
        print("%-40s GM=%2d  cost %.3e  max|f_Bn| %.3e  max|f_P| %.3e  [%.0fs]" % (f, GM, 0.5 * np.sum(r**2), np.abs(fu[:n]).max(), np.abs(fu[n:]).max(), time.time() - t), flush=True)
