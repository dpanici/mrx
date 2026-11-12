"""Evaluate DESC's free-boundary residual (BoundaryError: B.n and pressure balance, using the
same coil field) on the DESC and VMEC++ fixed- and free-boundary results.
VMEC++ wouts are loaded into DESC with VMECIO.load; |B| on the loaded LCFS is checked
against the wout's own |B| (bmnc extrapolated to s=1) to make sure the load is faithful."""
import os, sys, time
import numpy as np
import desc, desc.io
desc.config["avail_mem"] = float(os.environ.get("DESC_MEM", 6))
from desc.grid import LinearGrid
from desc.objectives import BoundaryError
from desc.vmec import VMECIO
from netCDF4 import Dataset
from common import load_field

field, src = load_field()
GM = int(os.environ.get("BGRID", 24))


def wout_modB_lcfs(path, grid):
    w = Dataset(path)
    xm, xn, b = w["xm_nyq"][:], w["xn_nyq"][:], w["bmnc"][:]
    bl = 1.5 * b[-1] - 0.5 * b[-2]  # half grid -> s = 1
    th, ze = grid.nodes[:, 1], grid.nodes[:, 2]
    # VMEC theta runs the opposite way to DESC theta (VMECIO flips it): theta_vmec = -theta_desc
    return np.cos(np.outer(-th, xm) - np.outer(ze, xn)) @ bl


def check(name, eq, wout=None):
    t = time.time()
    g = LinearGrid(M=GM, N=GM, NFP=eq.NFP)
    obj = BoundaryError(eq=eq, field=field, field_grid=src, field_fixed=True, source_grid=g, eval_grid=g,
                        bs_chunk_size=64, B_plasma_chunk_size=128)
    obj.build(verbose=0)
    print("== " + name, flush=True)
    obj.print_value(obj.xs(eq))
    line = ""
    if wout is not None:
        bd = np.asarray(eq.compute("|B|", grid=g)["|B|"])
        bw = wout_modB_lcfs(wout, g)
        line += " | |B| load check: max rel diff %.1e" % np.max(np.abs(bd - bw) / bw)
    print(line + "  [%.0fs]" % (time.time() - t), flush=True)


which = sys.argv[1:] or ["desc_fixed", "desc_free", "vmec_fixed", "vmec_free"]
for w in which:
    if w == "desc_fixed":
        check("DESC fixed (12,12,12)", desc.io.load("desc_fixed_L12_M12_N12_jacobi.h5"))
    elif w == "desc_free":
        check("DESC free (12,12,12)", desc.io.load("desc_free_L12_M12_N12.h5"))
    elif w.startswith("vmec"):
        kind = w.split("_")[1]
        tag = os.environ.get("VTAG", "m13_n12_ns101")
        path = "wout_vmec_%s_%s.nc" % (kind, tag)
        eq = VMECIO.load(path, L=int(os.environ.get("LLOAD", 24)), M=13, N=12, profile="iota")
        check("VMEC++ %s %s (loaded)" % (kind, tag), eq, path)
