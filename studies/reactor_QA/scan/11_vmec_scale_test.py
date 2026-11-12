"""VMEC++ convergence test of the current scale factor: python 11_vmec_scale_test.py SCALE
Runs (mpol, ntor, ns) = (9,8,101), (9,8,201), (13,12,201), (17,16,201) with current x SCALE and reports surface
distances to (17,16,201) at rho = 0.2..0.8, iota_max, and where iota crosses low-order rationals 2n/m (m <= 12)."""
import sys, time, numpy as np, vmecpp
from fractions import Fraction
from common import vmec_eval, compare, RHOS
sc = float(sys.argv[1])
res = {}
for mpol, ntor, ns in [(9, 8, 101), (9, 8, 201), (13, 12, 201), (17, 16, 201)]:
    inp = vmecpp.VmecInput.from_file("../input.reactor_QA_even")
    inp.ac = np.asarray(inp.ac) * sc; inp.curtor = inp.curtor * sc
    rbc, zbs = np.asarray(inp.rbc), np.asarray(inp.zbs); m0, n0 = rbc.shape[0], (rbc.shape[1] - 1) // 2
    R = np.zeros((mpol, 2 * ntor + 1)); Z = np.zeros_like(R); mm, nn = min(m0, mpol), min(n0, ntor)
    R[:mm, ntor - nn:ntor + nn + 1] = rbc[:mm, n0 - nn:n0 + nn + 1]; Z[:mm, ntor - nn:ntor + nn + 1] = zbs[:mm, n0 - nn:n0 + nn + 1]
    inp.mpol, inp.ntor, inp.rbc, inp.zbs = mpol, ntor, R, Z
    ax = lambda a: np.pad(np.asarray(a), (0, max(0, ntor + 1 - len(a))))[:ntor + 1]
    inp.raxis_c, inp.zaxis_s = ax(inp.raxis_c), ax(inp.zaxis_s)
    inp.ntheta, inp.nzeta = 4 * mpol, 4 * ntor
    inp.ns_array = np.array([51, ns]); inp.ftol_array = np.array([1e-10, 1e-12 if mpol > 13 else 1e-13])
    inp.niter_array = np.array([30000, 60000]); inp.return_outputs_even_if_not_converged = True
    t = time.time(); w = vmecpp.run(inp, verbose=0, max_threads=4).wout
    res[(mpol, ns)] = vmec_eval(w)
    print("scale %.2f %2d/%2d/%3d: %4.0fs fsq %.1e  iota_max %.4f" % (sc, mpol, ntor, ns, time.time() - t, w.fsqr + w.fsqz + w.fsql, np.abs(w.iotaf).max()), flush=True)
ref = res[(17, 201)]
for k, r in res.items():
    if k != (17, 201):
        c = compare(r, ref); print("   %s vs 17/16/201: rho .2-.8 [mm]" % (k,), np.round(1e3 * c["surf_dist"].max(0)[:-1], 1), "axis %.1f" % (1e3 * c["axis"]))
io = np.abs(ref["iota"]); s = ref["s"]
rats = sorted({Fraction(2 * n, m) for m in range(1, 13) for n in range(1, 13) if 0.3 < 2 * n / m < 0.7})
for q in rats:
    x = np.where(np.diff(np.sign(io - float(q))))[0]
    if len(x): print("   iota = %s (m=%d) at s = %s, shear |d iota/ds| = %s" % (q, q.denominator if q.numerator % 2 == 0 else 2 * q.denominator, np.round(s[x], 3), np.round(np.abs(np.gradient(io, s)[x]), 3)))
