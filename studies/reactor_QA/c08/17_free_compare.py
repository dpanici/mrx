"""Compare DESC and VMEC++ free-boundary results for the c08 case: python 17_free_compare.py [desc_free_tag]"""
import sys, pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import compare, PLANES

L = lambda t: pickle.load(open(t + ".pkl", "rb"))
dtag = sys.argv[1] if len(sys.argv) > 1 else "desc_free_L12_M12_N12"
D = L(dtag)
Dfix = L("desc_fixed_L12_M12_N12_jacobi")
V = {t: L("vmec_free_" + t) for t in ["m9_n8_ns51", "m9_n8_ns101", "m13_n12_ns101"]}
Vfix = L("vmec_fixed_m13_n12_ns101")


def row(name, a, b):
    c = compare(a, b)
    print("%-44s LCFS %6.1f | rho .2-.8 %s | axis %5.1f mm | max dIota %.4f (edge %.4f) | dV/V %.1e | dbeta/beta %.1e"
          % (name, 1e3 * c["lcfs"], np.round(1e3 * c["surf_dist"].max(0)[:-1], 1), 1e3 * c["axis"], c["iota"], c["iota_edge"], c["V"], c["beta"]))


print("== how far each free boundary moved from its own fixed-boundary start")
row("DESC free 12^3 vs DESC fixed 12^3", D, Dfix)
row("VMEC++ free 13/12/101 vs VMEC++ fixed 13/12/101", V["m13_n12_ns101"], Vfix)
print("== DESC free vs VMEC++ free")
for t, r in V.items():
    row("DESC free 12^3 vs VMEC++ free %s" % t, D, r)
print("== for reference: DESC fixed 12^3 vs VMEC++ fixed 13/12/101")
row("", Dfix, Vfix)
for name, r in [("DESC free", D), ("VMEC++ free 13/12/101", V["m13_n12_ns101"])]:
    io = np.abs(r["iota"])
    print("%-22s V %.2f m3  beta %.5f  iota(0.01) %.4f  iota_max %.4f  iota(1) %.4f" % (name, r["V"], r["beta"], np.interp(0.01, r["s"], io), io.max(), io[-1]))

fig, ax = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
for i in range(3):
    for r, col, ls, lab in [(Dfix, "0.55", ":", "fixed boundary (target of the coil fit)"), (D, "#1b56a3", "-", "DESC free (12,12,12)"),
                            (V["m13_n12_ns101"], "#b8461b", "--", "VMEC++ free 13/12, ns 101")]:
        for j in range(5):
            ax[i].plot(*r["surf"][i, j], color=col, ls=ls, lw=1.6 if j == 4 else 1.0, label=lab if j == 4 else None)
        ax[i].plot(r["axis_R"][i], r["axis_Z"][i], "x", color=col)
    ax[i].set_aspect("equal"); ax[i].set_title("φ = %.3f" % PLANES[i]); ax[i].set_xlabel("R [m]"); ax[i].set_ylabel("Z [m]")
ax[2].legend(fontsize=8, loc="upper left")
fig.savefig("free_desc_vs_vmec.png", dpi=110)
print("wrote free_desc_vs_vmec.png")
