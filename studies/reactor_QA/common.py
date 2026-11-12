"""Shared helpers: the external field, VMEC++ input, and evaluation of DESC / VMEC results on common ground."""
import pickle
import numpy as np

NFP = 2
PLANES = np.array([0.0, 0.25, 0.5]) * 2 * np.pi / NFP          # toroidal planes phi (half a field period)
RHOS = np.array([0.2, 0.4, 0.6, 0.8, 1.0])


def load_field():
    import desc.io
    from desc.grid import LinearGrid
    field = desc.io.load("coil_field.h5")
    g = pickle.load(open("coil_field_grid.pkl", "rb"))
    return field, LinearGrid(M=g["MSRC"], N=g["NSRC"], NFP=field.NFP)


def field_callback():
    """b_r, b_phi, b_z = f(r, phi, z) for VMEC++: the same discretised surface-current Biot-Savart sum DESC uses
    (same source grid), with the source positions and K dA computed once."""
    field, src = load_field()
    d = field.compute(["K", "x", "|e_theta x e_zeta|"], grid=src, basis="xyz")
    K, X = np.asarray(d["K"]), np.asarray(d["x"])
    dA = np.asarray(d["|e_theta x e_zeta|"]) * np.asarray(src.weights) / (4 * np.pi**2) * (2 * np.pi) * (2 * np.pi / field.NFP) * field.NFP
    KdA = K * dA[:, None] / field.NFP
    Xs, Ks = [], []
    for k in range(field.NFP):                                   # the other field periods, by rotation
        c, s_ = np.cos(2 * np.pi * k / field.NFP), np.sin(2 * np.pi * k / field.NFP)
        Rm = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        Xs.append(X @ Rm.T); Ks.append(KdA @ Rm.T)
    Xs, Ks = np.concatenate(Xs), np.concatenate(Ks)

    from desc.backend import jit, jnp
    Xj, Kj = jnp.asarray(Xs), jnp.asarray(Ks)

    @jit
    def _B(r, phi, z):
        x = jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi), z], axis=-1)
        dx = x[:, None, :] - Xj[None, :, :]
        B = 1e-7 * (jnp.cross(Kj[None], dx) / (jnp.sum(dx * dx, axis=-1) ** 1.5)[..., None]).sum(axis=1)
        return (B[:, 0] * jnp.cos(phi) + B[:, 1] * jnp.sin(phi), -B[:, 0] * jnp.sin(phi) + B[:, 1] * jnp.cos(phi), B[:, 2])

    def f(r, phi, z):
        shp = np.shape(r)
        b = _B(jnp.asarray(np.ravel(r)), jnp.asarray(np.ravel(phi)), jnp.asarray(np.ravel(z)))
        return tuple(np.asarray(c).reshape(shp) for c in b)
    return f


# ---------------------------------------------------------------- evaluation
def desc_eval(eq):
    """Surfaces, axis, profiles and scalars of a DESC equilibrium."""
    from desc.grid import LinearGrid, Grid
    th = np.linspace(0, 2 * np.pi, 361)
    surf = np.zeros((len(PLANES), len(RHOS), 2, len(th)))
    for i, ph in enumerate(PLANES):
        g = LinearGrid(rho=RHOS, theta=th, zeta=np.array([ph]), NFP=eq.NFP)
        d = eq.compute(["R", "Z"], grid=g)
        surf[i, :, 0] = np.asarray(d["R"]).reshape(g.num_zeta, g.num_rho, g.num_theta)[0] if False else \
            g.meshgrid_reshape(d["R"], "rtz")[:, :, 0]
        surf[i, :, 1] = g.meshgrid_reshape(d["Z"], "rtz")[:, :, 0]
    s = np.linspace(0, 1, 101)
    gp = LinearGrid(rho=np.sqrt(s[1:]), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    d = eq.compute(["iota", "current", "p"], grid=gp)
    ax = eq.compute(["R", "Z"], grid=LinearGrid(rho=np.array([0.0]), theta=np.array([0.0]), zeta=PLANES, NFP=eq.NFP))
    sc = eq.compute(["V", "<beta>_vol", "R0/a", "a", "R0", "<|B|>_vol", "W_B", "W_p", "<|F|>_vol",
                     "<|grad(|B|^2)|/2mu0>_vol"])
    return dict(surf=surf, s=s[1:], iota=np.asarray(gp.compress(d["iota"])), current=np.asarray(gp.compress(d["current"])),
                p=np.asarray(gp.compress(d["p"])), axis_R=np.asarray(ax["R"]), axis_Z=np.asarray(ax["Z"]),
                V=float(sc["V"]), beta=float(sc["<beta>_vol"]), aspect=float(sc["R0/a"]), a=float(sc["a"]),
                R0=float(sc["R0"]), Bavg=float(sc["<|B|>_vol"]), W_B=float(sc["W_B"]), W_p=float(sc["W_p"]),
                force=float(sc["<|F|>_vol"] / sc["<|grad(|B|^2)|/2mu0>_vol"]))


def vmec_eval(w):
    """The same quantities from a VMEC++ wout object."""
    th = np.linspace(0, 2 * np.pi, 361)
    xm, xn = np.asarray(w.xm), np.asarray(w.xn)
    rmnc, zmns = np.asarray(w.rmnc), np.asarray(w.zmns)
    if rmnc.shape[0] != len(xm):
        rmnc, zmns = rmnc.T, zmns.T                              # -> (mn, ns)
    ns = rmnc.shape[1]
    sfull = np.linspace(0, 1, ns)
    from scipy.interpolate import CubicSpline

    def at_s(c, s):
        """Mode coefficients at s: interpolate c/rho^m (smooth) in s, as m>0 modes behave like rho^m near the axis."""
        out = np.zeros((len(xm), len(s)))
        for k in range(len(xm)):
            m = int(xm[k])
            if m == 0:
                out[k] = CubicSpline(sfull, c[k])(s)
            else:
                w_ = np.sqrt(sfull[1:]) ** min(m, 2)
                y = c[k, 1:] / w_
                out[k] = CubicSpline(sfull[1:], y, extrapolate=True)(s) * np.sqrt(s) ** min(m, 2)
        return out
    rc, zs = at_s(rmnc, RHOS**2), at_s(zmns, RHOS**2)
    surf = np.zeros((len(PLANES), len(RHOS), 2, len(th)))
    for i, ph in enumerate(PLANES):
        ang = xm[:, None] * th[None, :] - xn[:, None] * ph
        surf[i, :, 0] = rc.T @ np.cos(ang)
        surf[i, :, 1] = zs.T @ np.sin(ang)
    axR = np.array([np.sum(rmnc[:, 0] * np.cos(-xn * ph)) for ph in PLANES])
    axZ = np.array([np.sum(zmns[:, 0] * np.sin(-xn * ph)) for ph in PLANES])
    s = np.linspace(0, 1, 101)[1:]
    mu0 = 4e-7 * np.pi
    return dict(surf=surf, s=s, iota=CubicSpline(sfull, np.asarray(w.iotaf))(s),
                p=CubicSpline(sfull, np.asarray(w.presf))(s), axis_R=axR, axis_Z=axZ,
                V=float(w.volume_p), beta=float(w.betatotal), aspect=float(w.aspect), a=float(w.Aminor_p),
                R0=float(w.Rmajor_p), Bavg=float(w.volavgB), W_B=float(w.wb) * 4 * np.pi**2 / mu0,
                W_p=float(w.wp) * 4 * np.pi**2 / mu0, fsq=float(w.fsqr + w.fsqz + w.fsql))


def curve_distance(A, B):
    """Max over the points of closed curve A (2, n) of the distance to the polyline B (2, m)."""
    P = A.T[:, None, :]
    Q0, Q1 = B.T[None, :-1, :], B.T[None, 1:, :]
    dq = Q1 - Q0
    t = np.clip(((P - Q0) * dq).sum(-1) / np.maximum((dq**2).sum(-1), 1e-300), 0, 1)
    dist = np.linalg.norm(P - (Q0 + t[..., None] * dq), axis=-1).min(axis=1)
    return dist.max()


def compare(a, b):
    """Differences between two evaluated results."""
    ds = np.array([[max(curve_distance(a["surf"][i, j], b["surf"][i, j]), curve_distance(b["surf"][i, j], a["surf"][i, j]))
                    for j in range(len(RHOS))] for i in range(len(PLANES))])
    return dict(surf_dist=ds, lcfs=ds[:, -1].max(), interior=ds[:, :-1].max(),
                axis=np.hypot(a["axis_R"] - b["axis_R"], a["axis_Z"] - b["axis_Z"]).max(),
                iota=np.abs(np.abs(a["iota"]) - np.abs(b["iota"])).max(), iota_edge=abs(abs(a["iota"][-1]) - abs(b["iota"][-1])),
                iota_axis=abs(abs(a["iota"][0]) - abs(b["iota"][0])),
                V=abs(a["V"] - b["V"]) / a["V"], beta=abs(a["beta"] - b["beta"]) / a["beta"])
