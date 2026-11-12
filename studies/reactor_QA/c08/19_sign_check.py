"""Direction check on the common fixed boundary: plasma B on the LCFS from the VMEC++ fixed wout
(computed directly from VMEC's own Fourier data, physical cylindrical components), from the DESC
fixed equilibrium, and the coil field there. Toroidal and poloidal parts are compared separately."""
import numpy as np
from netCDF4 import Dataset
import desc.io
from desc.grid import Grid
from common import field_callback

import sys
w = Dataset(sys.argv[1] if len(sys.argv) > 1 else "wout_vmec_fixed_m13_n12_ns101.nc"); w.set_auto_mask(False)
xm, xn, xmn, xnn = w["xm"][:], w["xn"][:], w["xm_nyq"][:], w["xn_nyq"][:]
rmnc, zmns = w["rmnc"][-1], w["zmns"][-1]
ext = lambda a: 1.5 * a[-1] - 0.5 * a[-2]
bu, bv = ext(w["bsupumnc"][:]), ext(w["bsupvmnc"][:])

u = np.linspace(0, 2 * np.pi, 16, endpoint=False)
v = np.linspace(0, np.pi, 5, endpoint=False)
U, Vv = [a.ravel() for a in np.meshgrid(u, v)]
A = np.outer(U, xm) - np.outer(Vv, xn)
An = np.outer(U, xmn) - np.outer(Vv, xnn)
R, Z = np.cos(A) @ rmnc, np.sin(A) @ zmns
Ru, Zu = -np.sin(A) @ (xm * rmnc), np.cos(A) @ (xm * zmns)
Rv, Zv = np.sin(A) @ (xn * rmnc), -np.cos(A) @ (xn * zmns)
Bsu, Bsv = np.cos(An) @ bu, np.cos(An) @ bv
# B = B^u e_u + B^v e_v, cylindrical (R, phi, Z) components; e_v has phi-component R
Bv_cyl = np.stack([Bsu * Ru + Bsv * Rv, Bsv * R, Bsu * Zu + Bsv * Zv], 1)

# DESC at the same physical points: DESC theta = pi - VMEC u on the shared boundary (VMECIO convention), check by position
eq = desc.io.load("desc_fixed_L12_M12_N12_jacobi.h5")
g = Grid(np.stack([np.ones_like(U), np.mod(np.pi - U, 2 * np.pi), Vv], 1), sort=False)
d = eq.compute(["R", "Z", "B_R", "B_phi", "B_Z"], grid=g)
print("position mismatch DESC(theta=pi-u) vs VMEC: max %.2e m" % np.max(np.hypot(d["R"] - R, d["Z"] - Z)))
Bd_cyl = np.stack([d["B_R"], d["B_phi"], d["B_Z"]], 1)

f = field_callback()
Bc = np.stack(f(R, Vv, Z), 1)

tor = lambda B: B[:, 1]
pol = lambda B: np.stack([B[:, 0], B[:, 2]], 1)
for name, B in [("VMEC++ plasma", Bv_cyl), ("DESC plasma", Bd_cyl), ("coil field", Bc)]:
    print("%-14s B_phi mean %+.3f T   |B_pol| mean %.3f T" % (name, tor(B).mean(), np.linalg.norm(pol(B), axis=1).mean()))
cos = lambda a, b: np.sum(a * b, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
print("cos(poloidal) VMEC vs DESC: min %.3f mean %.3f" % (cos(pol(Bv_cyl), pol(Bd_cyl)).min(), cos(pol(Bv_cyl), pol(Bd_cyl)).mean()))
print("cos(poloidal) VMEC vs coil: mean %.3f ; DESC vs coil: mean %.3f" % (cos(pol(Bv_cyl), pol(Bc)).mean(), cos(pol(Bd_cyl), pol(Bc)).mean()))
print("|B| VMEC vs DESC on LCFS: max rel diff %.2e" % np.max(np.abs(np.linalg.norm(Bv_cyl, axis=1) / np.linalg.norm(Bd_cyl, axis=1) - 1)))
