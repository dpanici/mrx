"""VMEC++ run of reactor_QA_even: python 03_vmec.py {fixed|free} MPOL NTOR NS [tag]"""
import os, sys, time, pickle
import numpy as np
import vmecpp
from common import field_callback, vmec_eval

mode, mpol, ntor, ns = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
tag = "vmec_%s_m%d_n%d_ns%d" % (mode, mpol, ntor, ns)
inp = vmecpp.VmecInput.from_file("input.reactor_QA_even")       # written by DESC's VMECIO from reactor_QA_even.h5
nss = [n for n in (25, 51, 99, 151, 201, 301) if n < ns] + [ns]
inp.ns_array = np.array(nss[-3:])
inp.ftol_array = np.array([1e-9, 1e-11, float(os.environ.get("FTOL", 1e-13))][-len(inp.ns_array):])
inp.niter_array = np.array([int(os.environ.get("NITER", 50000))] * len(inp.ns_array))
inp.return_outputs_even_if_not_converged = True
import os
inp.ntheta = int(os.environ.get("NTHETA", 4 * mpol))
inp.nzeta = int(os.environ.get("NZETA", 4 * ntor))
inp.delt = 0.7
rbc, zbs = np.asarray(inp.rbc), np.asarray(inp.zbs)             # (mpol0, 2*ntor0+1): resize to the new resolution
m0, n0 = rbc.shape[0], (rbc.shape[1] - 1) // 2
R = np.zeros((mpol, 2 * ntor + 1)); Z = np.zeros_like(R)
mm, nn = min(m0, mpol), min(n0, ntor)
R[:mm, ntor - nn:ntor + nn + 1] = rbc[:mm, n0 - nn:n0 + nn + 1]
Z[:mm, ntor - nn:ntor + nn + 1] = zbs[:mm, n0 - nn:n0 + nn + 1]
inp.mpol, inp.ntor, inp.rbc, inp.zbs = mpol, ntor, R, Z
ax = lambda a: np.pad(np.asarray(a), (0, max(0, ntor + 1 - len(a))))[:ntor + 1]
inp.raxis_c, inp.zaxis_s = ax(inp.raxis_c), ax(inp.zaxis_s)
kw = {}
if mode == "free":
    inp.lfreeb = True
    inp.nvacskip = 6
    kw["magnetic_field"] = field_callback()
t = time.time()
out = vmecpp.run(inp, verbose=1, max_threads=4, **kw)
w = out.wout
print("%s: %.0fs  fsqr %.2e fsqz %.2e fsql %.2e  iter %s" % (tag, time.time() - t, w.fsqr, w.fsqz, w.fsql, getattr(w, "itfsq", "?")))
w.save("wout_%s.nc" % tag)
r = vmec_eval(w); r["wall"] = time.time() - t
pickle.dump(r, open(tag + ".pkl", "wb"))
print({k: v for k, v in r.items() if np.isscalar(v)}, "iota", r["iota"][[0, -1]])
