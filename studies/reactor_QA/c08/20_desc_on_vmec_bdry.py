"""DESC fixed-boundary solve (12,12,12) on the VMEC++ free boundary, then the same free-boundary
residual (BoundaryError, same coil field and grids) as used for the DESC free run.
VMEC u maps to DESC theta = pi - u for these equilibria (checked on the shared fixed boundary)."""
import os, sys, time
import numpy as np
import desc, desc.io
desc.config["avail_mem"] = float(os.environ.get("DESC_MEM", 6))
from netCDF4 import Dataset
from desc.geometry import FourierRZToroidalSurface
from desc.grid import Grid, LinearGrid
from desc.objectives import BoundaryError
from desc.vmec_utils import ptolemy_identity_fwd
from common import load_field

vtag = sys.argv[1] if len(sys.argv) > 1 else "vmec_free_m13_n12_ns101"
start = sys.argv[2] if len(sys.argv) > 2 else "desc_free_L12_M12_N12.h5"
out = "desc_on_%s" % vtag
w = Dataset("wout_%s.nc" % vtag); w.set_auto_mask(False)
NFP = int(w["nfp"][:])
xm, xn = w["xm"][:], w["xn"][:]
m, n, Rb = ptolemy_identity_fwd(xm, xn / NFP, s=np.zeros_like(xm, float), c=w["rmnc"][-1])
_, _, Zb = ptolemy_identity_fwd(xm, xn / NFP, s=w["zmns"][-1], c=np.zeros_like(xm, float))
Rb, Zb, m, n = np.ravel(Rb), np.ravel(Zb), np.ravel(m), np.ravel(n)
# that surface has theta = u; reparametrise theta -> pi - theta:
# cos(m(pi - t)) = (-1)^m cos(mt) (DESC m >= 0), sin(|m|(pi - t)) = -(-1)^m sin(|m|t) (DESC m < 0)
fac = np.where(m >= 0, 1.0, -1.0) * (-1.0) ** np.abs(m)
Rb, Zb = fac * Rb, fac * Zb
modes = np.stack([m, n], 1).astype(int)
surf = FourierRZToroidalSurface(Rb, Zb, modes, modes, NFP, sym=True, check_orientation=False)

eq = desc.io.load(start)
surf.change_resolution(M=eq.M, N=eq.N)
assert np.all(surf.R_basis.modes == eq.surface.R_basis.modes) and np.all(surf.Z_basis.modes == eq.surface.Z_basis.modes)
# check the mapping against VMEC's own LCFS
u = np.linspace(0, 2 * np.pi, 24, endpoint=False); v = np.linspace(0, np.pi, 7, endpoint=False)
U, V = [a.ravel() for a in np.meshgrid(u, v)]
A = np.outer(U, xm) - np.outer(V, xn)
Rw, Zw = np.cos(A) @ w["rmnc"][-1], np.sin(A) @ w["zmns"][-1]
d = surf.compute(["R", "Z"], grid=Grid(np.stack([np.ones_like(U), np.mod(np.pi - U, 2 * np.pi), V], 1), sort=False))
print("boundary mapping check: max position mismatch %.1e m" % np.max(np.hypot(d["R"] - Rw, d["Z"] - Zw)), flush=True)

t = time.time()
eq.perturb(deltas={"Rb_lmn": surf.R_lmn - eq.Rb_lmn, "Zb_lmn": surf.Z_lmn - eq.Zb_lmn}, order=1, verbose=1, copy=False)
print("perturb %.0fs" % (time.time() - t), flush=True)
t = time.time()
eq.solve(verbose=1, ftol=1e-8, xtol=1e-10, gtol=1e-10, maxiter=int(os.environ.get("MAXITER", 100)), copy=False)
print("solve %.0fs" % (time.time() - t), flush=True)
eq.save(out + ".h5")

field, src = load_field()
g = LinearGrid(M=24, N=24, NFP=eq.NFP)
obj = BoundaryError(eq=eq, field=field, field_grid=src, field_fixed=True, source_grid=g, eval_grid=g,
                    bs_chunk_size=64, B_plasma_chunk_size=128)
obj.build(verbose=0)
print("== free-boundary residual of DESC's equilibrium on the %s boundary" % vtag)
obj.print_value(obj.xs(eq))

# |B| on the LCFS: DESC on this boundary vs the VMEC++ wout itself
bl = 1.5 * w["bmnc"][-1] - 0.5 * w["bmnc"][-2]
Bw = np.cos(np.outer(U, w["xm_nyq"][:]) - np.outer(V, w["xn_nyq"][:])) @ bl
Bd = np.asarray(eq.compute("|B|", grid=Grid(np.stack([np.ones_like(U), np.mod(np.pi - U, 2 * np.pi), V], 1), sort=False))["|B|"])
print("|B| on LCFS, DESC vs VMEC++ wout: max rel diff %.2e" % np.max(np.abs(Bd / Bw - 1)))
