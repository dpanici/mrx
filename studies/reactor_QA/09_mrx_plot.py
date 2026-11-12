"""MRX fixed-boundary relaxation of each DESC / VMEC++ equilibrium against the input's nested surfaces.

python 09_mrx_plot.py TAG RESULT_PKL [TAG RESULT_PKL ...]

Per run, mrx_<TAG>.png: Poincare sections of the MRX initial field (top row, B = dA' of the input) and of the
relaxed field (bottom row), over the input equilibrium's flux surfaces rho = 0.2 ... 1 and axis, at phi = 0, 1/4 and
1/2 of a field period. mrx_summary.png: the departure of every relaxed field line from the flux surface it was
started on, max |r - r0| over its crossings (r = sqrt(s) the input's radial label, which is MRX's logical r), against
r0 and against the line's iota. The table gives the same numbers.
"""
import sys, json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import PLANES, RHOS

ZETAS = (0, 0.25, 0.5)


def departure(tr, f):
    """max |logr - r0| of each line over its crossings in all three planes."""
    r0 = tr[f + "_seed_r"]
    d = np.zeros_like(r0)
    for z in ZETAS:
        d = np.fmax(d, np.nanmax(np.abs(tr["%s_zeta%g_logr" % (f, z)] - r0[:, None]), axis=1))
    return d


pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))
sfig, sax = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
print("%-22s %10s %10s %7s %9s %9s %8s %18s %10s" % ("run", "F_0", "F_best", "steps", "axis[mm]", "dep_init",
                                                     "dep_max", "at r0 (iota)", "chaotic"))
for c, (tag, pk) in enumerate(pairs):
    eq = pickle.load(open(pk, "rb"))
    tr = np.load("mrx_%s/trace.npz" % tag)
    js = json.load(open("mrx_%s/relax.json" % tag))
    names = [str(f) for f in tr["fields"]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    shift = 0.0
    for row, (f, col, lab) in enumerate(zip(names, ["0.35", "C3"], ["MRX initial field", "MRX relaxed field"])):
        keep = tr[f + "_keep"]
        for i, z in enumerate(ZETAS):
            ax, key = axes[row, i], "%s_zeta%g" % (f, z)
            ax.plot(tr[key + "_R"][keep].ravel(), tr[key + "_Z"][keep].ravel(), ".", ms=0.5, color=col, label=lab)
            for j in range(len(RHOS)):
                ax.plot(*eq["surf"][i, j], "-", color="C0", lw=1.0, label="input nested surfaces" if j == 0 else None)
            ax.plot(eq["axis_R"][i], eq["axis_Z"][i], "x", color="C0", ms=8)
            aR, aZ = float(np.ravel(tr[key + "_axisR"])[0]), float(np.ravel(tr[key + "_axisZ"])[0])
            ax.plot(aR, aZ, "+", color=col, ms=10)
            if row == 1:
                shift = max(shift, np.hypot(aR - eq["axis_R"][i], aZ - eq["axis_Z"][i]))
            ax.set_aspect("equal"); ax.set_title("%s: %s, phi = %.3f" % (tag, lab, PLANES[i]), fontsize=9)
            ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
        axes[row, 0].legend(fontsize=7, markerscale=10, loc="lower left")
    fig.savefig("mrx_%s.png" % tag, dpi=100)
    plt.close(fig)
    d0, d1 = departure(tr, names[0]), departure(tr, names[1])
    keep, r0, io = tr[names[1] + "_keep"], tr[names[1] + "_seed_r"], np.abs(tr[names[1] + "_iota"])
    k = np.nanargmax(np.where(keep, d1, -1))
    print("%-22s %10.3e %10.3e %7d %9.1f %9.1e %8.4f %9.3f (%6.4f) %10d" % (
        tag, js["trace"]["F"][0], min(js["trace"]["F"]), js["summary"]["steps"], 1e3 * shift,
        np.nanmax(d0[tr[names[0] + "_keep"]]), d1[k], r0[k], io[k], int((keep & tr[names[1] + "_chaotic"]).sum())))
    ls = "-" if tag.startswith("desc") else "--"
    o = np.argsort(r0)
    sax[0].plot(r0[o], d1[o], ls, marker=".", color="C%d" % c, label=tag)
    sax[1].plot(io[keep], d1[keep], ".", color="C%d" % c, label=tag)
sax[0].set_xlabel("r0 = sqrt(s) of the line's start surface"); sax[0].set_ylabel("max |r - r0| along the relaxed line")
sax[1].set_xlabel("|iota| of the relaxed line"); sax[1].axvline(2 / 3, color="k", lw=0.8, ls=":")
sax[1].text(2 / 3, sax[1].get_ylim()[1] * 0.9, " 2/3", fontsize=8)
sax[0].legend(fontsize=7)
sfig.savefig("mrx_summary.png", dpi=110)
