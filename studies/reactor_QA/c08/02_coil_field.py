"""Step 2: an external (coil) field for the free-boundary runs.

reactor_QA has no coils, so a REGCOIL current potential on a winding surface offset from the plasma is solved for
the field that, together with the plasma's own field (virtual casing), has B.n = 0 on the fixed-boundary LCFS.
The same object (same source grid) is then used by DESC (BoundaryError) and by VMEC++ (field callback).
"""
import pickle, time
import numpy as np
import desc.io
from desc.grid import LinearGrid
from desc.magnetic_fields import FourierCurrentPotentialField, solve_regularized_surface_current

OFFSET, MPHI, NPHI, MSRC, NSRC = 1.7, 10, 10, 36, 36
eq = desc.io.load("desc_fixed_L12_M12_N12_jacobi.h5")
t = time.time()
ws = eq.surface.constant_offset_surface(offset=OFFSET, M=12, N=12, grid=LinearGrid(M=24, N=24, NFP=eq.NFP))
field = FourierCurrentPotentialField.from_surface(ws, M_Phi=MPHI, N_Phi=NPHI)
src = LinearGrid(M=MSRC, N=NSRC, NFP=eq.NFP)
ev = LinearGrid(M=24, N=24, NFP=eq.NFP)
fields, data = solve_regularized_surface_current(
    field, eq, lambda_regularization=np.array([1e-22, 1e-20, 1e-18]), current_helicity=(1, 0), vacuum=False,
    source_grid=src, eval_grid=ev, vc_source_grid=LinearGrid(M=48, N=48, NFP=eq.NFP), verbose=1, chunk_size=20, B_plasma_chunk_size=200)
Bavg = float(eq.compute("<|B|>_vol")["<|B|>_vol"])
for lam, chi2B, chi2K in zip(data["lambda_regularization"], data["chi^2_B"], data["chi^2_K"]):
    print("lambda %.0e  chi2_B %.3e  chi2_K %.3e" % (lam, chi2B, chi2K))
field = fields[1]
Bn = np.asarray(data["Bn_total"][1])
print("chosen lambda=1e-20: max|B.n| / <B> = %.3e, rms %.3e   (%.0fs)" % (np.abs(Bn).max() / Bavg, np.sqrt((Bn**2).mean()) / Bavg, time.time() - t))
print("G = %.4e A, I = %.4e A" % (field.G, field.I))
field.save("coil_field.h5")
pickle.dump(dict(MSRC=MSRC, NSRC=NSRC), open("coil_field_grid.pkl", "wb"))
