"""Compare every DESC and VMEC++ result found in this directory: python 06_compare.py {fixed|free} [figure.png]"""
import sys, glob, pickle, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import compare, PLANES, RHOS

mode = sys.argv[1]
runs = {}
for f in sorted(glob.glob("desc_%s_L*.pkl" % mode)) + sorted(glob.glob("vmec_%s_m*.pkl" % mode)):
    runs[f[:-4]] = pickle.load(open(f, "rb"))
def reskey(k):
    n = [int(x) for x in re.findall(r"[LMNmn](\d+)|ns(\d+)", k) for x in x if x]
    return (n[0] * n[1] * n[2], n[0], "dna" in k) if k.startswith("desc") else (n[2], n[0] * n[1])
desc = sorted([k for k in runs if k.startswith("desc")], key=reskey)
vmec = sorted([k for k in runs if k.startswith("vmec")], key=reskey)
runs = {k: runs[k] for k in desc + vmec}

def at(r, s0):
    return np.interp(s0, r["s"], np.abs(r["iota"]))

print("%-40s %8s %8s %8s %8s %10s %9s %8s" % ("run", "iota.01", "iota.25", "iota.5", "iota1", "beta", "axisR0", "wall"))
for k, r in runs.items():
    print("%-40s %8.4f %8.4f %8.4f %8.4f %10.6f %9.5f %8.0f" % (k, at(r, .01), at(r, .25), at(r, .5), at(r, 1), r["beta"],
                                                             r["axis_R"][0], r.get("wall", np.nan)))
print("\nDESC vs VMEC++: max distance between flux surfaces (rho=%s) [mm], axis [mm], max |d iota|" % RHOS[:-1])
print("%-36s" % "" + "".join("%-26s" % v.replace("vmec_%s_" % mode, "") for v in vmec))
table = {}
for d in desc:
    row = []
    for v in vmec:
        c = compare(runs[d], runs[v]); table[(d, v)] = c
        row.append("%6.2f %6.2f %8.1e    " % (1e3 * c["interior"], 1e3 * c["axis"], c["iota"]))
    print("%-36s" % d.replace("desc_%s_" % mode, "") + "".join(row))
if mode == "free":
    print("\nLCFS distance [mm]:")
    for (d, v), c in table.items():
        print("  %-36s %-26s %7.2f" % (d, v, 1e3 * c["lcfs"]))
print("\nself-convergence (each code against its own highest resolution):")
for grp in (desc, vmec):
    for k in grp[:-1] if len(grp) > 1 else []:
        c = compare(runs[k], runs[grp[-1]])
        print("  %-36s vs %-36s interior %6.2f mm  axis %6.2f mm  iota %.1e" % (k, grp[-1], 1e3 * c["interior"], 1e3 * c["axis"], c["iota"]))

if len(sys.argv) > 2:
    sel = sys.argv[3].split(",") if len(sys.argv) > 3 else [desc[-1], vmec[-1]]
    fig, ax = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(runs)))
    for c, (k, r) in zip(cols, runs.items()):
        ax[0].plot(r["s"], np.abs(r["iota"]), color=c, ls="-" if k.startswith("desc") else "--", label=k)
    ax[0].set_xlabel("s"); ax[0].set_ylabel("|iota|"); ax[0].legend(fontsize=6); ax[0].set_title("rotational transform")
    for i in range(3):
        for k, ls, col in zip(sel, ["-", "--"], ["C0", "C3"]):
            for j in range(len(RHOS)):
                ax[i + 1].plot(*runs[k]["surf"][i, j], ls=ls, color=col, lw=1, label=k if j == 0 else None)
            ax[i + 1].plot(runs[k]["axis_R"][i], runs[k]["axis_Z"][i], "x" if ls == "-" else "+", color=col)
        ax[i + 1].set_aspect("equal"); ax[i + 1].set_title("phi = %.2f" % PLANES[i]); ax[i + 1].set_xlabel("R [m]")
    ax[1].legend(fontsize=7); ax[1].set_ylabel("Z [m]")
    fig.savefig(sys.argv[2], dpi=110)
