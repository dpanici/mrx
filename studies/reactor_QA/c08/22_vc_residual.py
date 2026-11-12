"""Free-boundary residual of a VMEC++ result computed from VMEC++'s own LCFS data, with DESC's
singular virtual-casing integral and the same coil field as DESC's BoundaryError:
    B_out = B_coil + B_vc[n x B_in] + B_in/2,   residuals  B_out.n  and  |B_in|^2 + 2 mu0 p - |B_out|^2,
weighted by |e_theta x e_zeta| as in BoundaryError.  Geometry comes from the wout's LCFS
(as a DESC surface, theta = pi - u so it is right-handed), B_in from bsupu/bsupv extrapolated to s=1.
Usage: python 22_vc_residual.py wout1.nc [wout2.nc ...]   ('desc:FILE.h5' evaluates a DESC equilibrium the same way)"""
import os, sys, json
import numpy as np
import desc, desc.io
desc.config["avail_mem"] = 6.0
from netCDF4 import Dataset
from desc.geometry import FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.integrals.singularities import FFTInterpolator, best_params, best_ratio, virtual_casing_biot_savart
from desc.vmec_utils import ptolemy_identity_fwd
from common import load_field

field, src = load_field()
GMS = [int(x) for x in os.environ.get("GMS", "32,48").split(",")]


def vmec_surface(w):
    NFP = int(w["nfp"][:]); xm, xn = w["xm"][:], w["xn"][:]
    m, n, Rb = ptolemy_identity_fwd(xm, xn / NFP, s=np.zeros_like(xm, float), c=w["rmnc"][-1])
    _, _, Zb = ptolemy_identity_fwd(xm, xn / NFP, s=w["zmns"][-1], c=np.zeros_like(xm, float))
    Rb, Zb, m, n = np.ravel(Rb), np.ravel(Zb), np.ravel(m), np.ravel(n)
    fac = np.where(m >= 0, 1.0, -1.0) * (-1.0) ** np.abs(m)    # theta -> pi - theta
    modes = np.stack([m, n], 1).astype(int)
    return FourierRZToroidalSurface(fac * Rb, fac * Zb, modes, modes, NFP, sym=True, check_orientation=False)


def vmec_B(w, theta, zeta):
    """B (R, phi, Z components) on the LCFS at DESC angles (theta, zeta), u = pi - theta, v = zeta."""
    xm, xn, xmn, xnn = w["xm"][:], w["xn"][:], w["xm_nyq"][:], w["xn_nyq"][:]
    r, z = w["rmnc"][-1], w["zmns"][-1]
    ext = lambda a: 1.5 * a[-1] - 0.5 * a[-2]
    bu, bv = ext(w["bsupumnc"][:]), ext(w["bsupvmnc"][:])
    U, V = np.pi - theta, zeta
    A, An = np.outer(U, xm) - np.outer(V, xn), np.outer(U, xmn) - np.outer(V, xnn)
    R, Z = np.cos(A) @ r, np.sin(A) @ z
    Ru, Zu = -np.sin(A) @ (xm * r), np.cos(A) @ (xm * z)
    Rv, Zv = np.sin(A) @ (xn * r), -np.cos(A) @ (xn * z)
    Bu, Bv = np.cos(An) @ bu, np.cos(An) @ bv
    return np.stack([Bu * Ru + Bv * Rv, Bv * R, Bu * Zu + Bv * Zv], 1), R, Z, np.interp(1.0, np.linspace(0, 1, len(w["presf"][:])), w["presf"][:])


def residual(surf_or_eq, B_fun, GM, label):
    g = LinearGrid(M=GM, N=GM, NFP=surf_or_eq.NFP)
    keys = ["R", "phi", "Z", "n_rho", "|e_theta x e_zeta|", "e_theta", "e_zeta"]
    if isinstance(surf_or_eq, FourierRZToroidalSurface):
        data = surf_or_eq.compute(keys, grid=g)
    else:
        data = surf_or_eq.compute(keys + ["B", "p"], grid=g)
    data = {k: np.asarray(v) for k, v in data.items()}
    B, p, pos = B_fun(g, data)
    data["B"], data["K_vc"] = B, np.cross(data["n_rho"], B) / (4e-7 * np.pi)
    st, sz, q = best_params(g, best_ratio(data))
    interp = FFTInterpolator(g, g, st, sz, q)
    Bout = np.asarray(virtual_casing_biot_savart(data, data, interp, chunk_size=64)) + B / 2
    x = np.stack([data["R"], data["phi"], data["Z"]], 1)
    Bout = Bout + np.asarray(field.compute_magnetic_field(x, source_grid=src, basis="rpz", chunk_size=64))
    a = data["|e_theta x e_zeta|"]
    Bn = np.sum(Bout * data["n_rho"], 1)
    dP = (np.sum(B * B, 1) + 2 * 4e-7 * np.pi * p - np.sum(Bout * Bout, 1))
    wq = g.weights / g.weights.sum() * 4 * np.pi**2 / g.NFP          # quadrature weights in (theta, zeta)
    Bsq = np.sum(B * B, 1)
    out = dict(rms_Bn=np.sqrt(np.sum(wq * a * Bn**2) / np.sum(wq * a)) / np.sqrt(np.mean(Bsq)),
               max_Bn=np.max(np.abs(Bn)) / np.sqrt(np.mean(Bsq)),
               rms_P=np.sqrt(np.sum(wq * a * dP**2) / np.sum(wq * a)) / np.mean(Bsq),
               max_P=np.max(np.abs(dP)) / np.mean(Bsq))
    print("%-48s GM=%2d  B.n/|B|: rms %.2e max %.2e | (B_in^2+2mu0p-B_out^2)/B^2: rms %.2e max %.2e%s" % (
        label, GM, out["rms_Bn"], out["max_Bn"], out["rms_P"], out["max_P"], pos), flush=True)
    return out


RES = {}
for arg in sys.argv[1:]:
    for GM in GMS:
        if arg.startswith("desc:"):
            eq = desc.io.load(arg[5:])
            def B_fun(g, data):
                return data["B"], data["p"], ""
            RES.setdefault(arg[5:].replace(".h5", ""), {})[GM] = residual(eq, B_fun, GM, arg[5:].replace(".h5", ""))
        else:
            w = Dataset(arg); w.set_auto_mask(False)
            surf = vmec_surface(w)
            def B_fun(g, data, w=w):
                B, R, Z, pe = vmec_B(w, g.nodes[:, 1], g.nodes[:, 2])
                mis = np.max(np.hypot(R - data["R"], Z - data["Z"]))
                return B, pe, "  (pos. check %.0e m)" % mis
            RES.setdefault(arg.replace("wout_", "").replace(".nc", ""), {})[GM] = residual(surf, B_fun, GM, arg.replace("wout_", "").replace(".nc", ""))

if os.environ.get("VC_JSON"):
    p = os.environ["VC_JSON"]
    old = json.load(open(p)) if os.path.exists(p) else {}
    for k, v in RES.items():
        old[k] = {str(g): {a: float(b) for a, b in o.items()} for g, o in v.items()}
    json.dump(old, open(p, "w"), indent=1)
