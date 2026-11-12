"""Figures for the write-up (report/*.png) from the saved results of both cases."""
import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RHOS, PLANES

plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2a28", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                     "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False})
BLUE = ["#9cc3f0", "#5f9be3", "#2a78d6", "#1b56a3", "#0f3770"]
ORNG = ["#f6b89c", "#f08d62", "#eb6834", "#b8461b", "#7d2c0e"]
INK = "#2b2a28"
L = lambda path: pickle.load(open(path + ".pkl", "rb"))


def iota_panel(ax, runs, title, marks):
    for lab, r, col, ls in runs:
        ax.plot(r["s"], np.abs(r["iota"]), color=col, lw=1.6, ls=ls, label=lab)
    for q, t in marks:
        ax.axhline(q, color="#8a8984", lw=0.8, ls=":")
        ax.text(1.0, q, " " + t, fontsize=8, va="center", ha="left", color="#52514e", clip_on=False)
    ax.set_xlabel("s (normalised toroidal flux)"); ax.set_ylabel("|ι|"); ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.set_xlim(0, 1); ax.legend(fontsize=7.5, loc="lower right")


def surf_panel(ax, runs, plane, title):
    for lab, r, col, ls in runs:
        for j in range(len(RHOS)):
            ax.plot(*r["surf"][plane, j], color=col, lw=1.2 if ls == "-" else 1.4, ls=ls, label=lab if j == 0 else None)
        ax.plot(r["axis_R"][plane], r["axis_Z"][plane], "x" if ls == "-" else "+", color=col, ms=7)
    ax.set_aspect("equal"); ax.grid(False); ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
    ax.set_title(title, fontsize=10, loc="left", color=INK)


# ---- original even-current case
o = lambda t: L("reactor_QA_study_dummy") if False else L(t)
orig_desc = [("DESC (8,8,8)", L("desc_fixed_L8_M8_N8_jacobi"), BLUE[1], "-"), ("DESC (12,12,12)", L("desc_fixed_L12_M12_N12_jacobi"), BLUE[3], "-")]
orig_vmec = [("VMEC++ 9/8, ns 51", L("vmec_fixed_m9_n8_ns51"), ORNG[0], "--"), ("VMEC++ 9/8, ns 101", L("vmec_fixed_m9_n8_ns101"), ORNG[1], "--"),
             ("VMEC++ 9/8, ns 201", L("vmec_fixed_m9_n8_ns201"), ORNG[2], "--"), ("VMEC++ 9/8, ns 401", L("vmec_fixed_m9_n8_ns401"), ORNG[3], "--"),
             ("VMEC++ 17/16, ns 201", L("vmec_fixed_m17_n16_ns201"), ORNG[4], "--")]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
iota_panel(ax[0], orig_desc + orig_vmec, "Rotational transform", [(2 / 3, "2/3")])
ax[0].set_ylim(0.3, 0.7)
ins = ax[0].inset_axes([0.42, 0.42, 0.33, 0.3]); [ins.plot(r["s"], np.abs(r["iota"]), color=c, ls=ls, lw=1.2) for _, r, c, ls in orig_desc + orig_vmec]
ins.axhline(2 / 3, color="#8a8984", lw=0.8, ls=":"); ins.set_xlim(0.55, 0.95); ins.set_ylim(0.635, 0.672); ins.tick_params(labelsize=7); ins.set_title("near the peak", fontsize=7)
surf_panel(ax[1], [orig_desc[1], orig_vmec[-1]], 2, "Flux surfaces, φ = half field period")
surf_panel(ax[2], [orig_desc[1], orig_vmec[-1]], 1, "Flux surfaces, φ = quarter field period")
ax[1].legend(fontsize=7.5, loc="upper left")
fig.savefig("report/orig_fixed.png", dpi=130)

# ---- c08 case
c = lambda t: L("c08/" + t)
c08_desc = [("DESC (12,12,12)", c("desc_fixed_L12_M12_N12_jacobi"), BLUE[0], "-"), ("DESC (16,16,16)", c("desc_fixed_L16_M16_N16_jacobi"), BLUE[2], "-"),
            ("DESC (20,12,12)", c("desc_fixed_L20_M12_N12_jacobi"), BLUE[4], "-")]
import os
if os.path.exists("c08/desc_fixed_L20_M16_N16_jacobi.pkl"):
    c08_desc.append(("DESC (20,16,16)", c("desc_fixed_L20_M16_N16_jacobi"), "#081f45", "-"))
c08_vmec = [("VMEC++ 9/8, ns 51", c("vmec_fixed_m9_n8_ns51"), ORNG[0], "--"), ("VMEC++ 9/8, ns 201", c("vmec_fixed_m9_n8_ns201"), ORNG[2], "--"),
            ("VMEC++ 21/20, ns 201", c("vmec_fixed_m21_n20_ns201"), ORNG[4], "--")]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
iota_panel(ax[0], c08_desc + c08_vmec, "Rotational transform", [(2 / 3, "2/3"), (0.6, "3/5"), (4 / 7, "4/7"), (0.5, "1/2"), (0.4, "2/5")])
ax[0].set_ylim(0.3, 0.7)
best = [c08_desc[-1], c08_vmec[-1]]
surf_panel(ax[1], best, 2, "Flux surfaces, φ = half field period")
surf_panel(ax[2], best, 1, "Flux surfaces, φ = quarter field period")
ax[1].legend(fontsize=7.5, loc="upper left")
fig.savefig("report/c08_fixed.png", dpi=130)

# ---- MRX departure summaries
def dep(tr, f="best"):
    r0 = tr[f + "_seed_r"]; d = np.zeros_like(r0)
    for z in (0, 0.25, 0.5):
        d = np.fmax(d, np.nanmax(np.abs(tr["%s_zeta%g_logr" % (f, z)] - r0[:, None]), axis=1))
    return r0, d, np.abs(tr[f + "_iota"]), tr[f + "_keep"]


def mrx_panel(axs, groups, title, marks):
    for runs, cols, ls in groups:
        for (lab, path), col in zip(runs, cols):
            tr = np.load(path + "/trace.npz"); r0, d, io, keep = dep(tr); o_ = np.argsort(r0)
            axs[0].plot(r0[o_], d[o_], color=col, lw=1.3, ls=ls, marker=".", ms=3, label=lab)
            axs[1].plot(io[keep], d[keep], ".", ms=4, color=col)
    for q, t in marks:
        axs[1].axvline(q, color="#8a8984", lw=0.8, ls=":"); axs[1].text(q, 0.515, t, rotation=90, fontsize=7, va="top", ha="right", color="#52514e")
    axs[0].set_xlabel("r₀ = √s of the surface the line starts on"); axs[0].set_ylabel("max |r − r₀| along the relaxed line")
    axs[1].set_xlabel("|ι| of the relaxed line"); axs[0].set_title(title, fontsize=10, loc="left", color=INK)
    for a in axs: a.set_ylim(-0.01, 0.53)
    axs[0].legend(fontsize=7, ncol=2, loc="upper left")


fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), constrained_layout=True)
mrx_panel(ax, [([("DESC (8,8,8)", "mrx_desc_L8_jacobi"), ("DESC (12,12,12)", "mrx_desc_L12_jacobi")], [BLUE[1], BLUE[3]], "-"),
               ([("VMEC++ 9/8, ns 51", "mrx_vmec_m9_ns51"), ("VMEC++ 9/8, ns 101", "mrx_vmec_m9_ns101"), ("VMEC++ 13/12, ns 101", "mrx_vmec_m13_ns101")],
                [ORNG[0], ORNG[2], ORNG[4]], "--")],
          "Original case: MRX (12,16,16) relaxation of each fixed-boundary equilibrium", [(2 / 3, "2/3"), (0.5, "1/2"), (0.4, "2/5")])
fig.savefig("report/orig_mrx.png", dpi=130)

fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), constrained_layout=True)
mrx_panel(ax, [([("DESC (8,8,8)", "c08/mrx_desc_L8_jacobi"), ("DESC (12,12,12)", "c08/mrx_desc_L12_jacobi"), ("DESC (16,16,16)", "c08/mrx_desc_L16_jacobi"),
                 ("DESC (16,16,16) · MRX (16,32,32)", "c08/mrx_desc_L16_jacobi_wout_r32")],
                [BLUE[0], BLUE[2], BLUE[4], "#081f45"], "-"),
               ([("VMEC++ 9/8, ns 51", "c08/mrx_vmec_m9_ns51"), ("VMEC++ 9/8, ns 201", "c08/mrx_vmec_m9_ns201"), ("VMEC++ 9/8, ns 401", "c08/mrx_vmec_m9_ns401")],
                ["#9fd9b9", "#1baf7a", "#0b6b49"], "--"),
               ([("VMEC++ 13/12, ns 201", "c08/mrx_vmec_m13_ns201"), ("VMEC++ 17/16, ns 201", "c08/mrx_vmec_m17_ns201"), ("VMEC++ 21/20, ns 201", "c08/mrx_vmec_m21_ns201"),
                 ("VMEC++ 21/20, ns 201 · MRX (16,32,32)", "c08/mrx_vmec_m21_ns201_r32")], [ORNG[1], ORNG[2], ORNG[3], "#3b1406"], "--")],
          "c08 case: MRX relaxation of each fixed-boundary equilibrium (MRX 12,16,16 unless noted)", [(0.6, "3/5"), (4 / 7, "4/7"), (0.5, "1/2"), (0.4, "2/5")])
fig.savefig("report/c08_mrx.png", dpi=130)
# ---- c08 free boundary: DESC free vs VMEC++ free against the fixed boundary
import os, json
from netCDF4 import Dataset
fx = c("desc_fixed_L12_M12_N12_jacobi"); dfree = c("desc_free_L12_M12_N12"); f13 = c("vmec_free_m13_n12_ns101")
fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
for i, (pl, t) in enumerate([(0, "φ = 0"), (1, "φ = quarter field period"), (2, "φ = half field period")]):
    surf_panel(ax[i], [("DESC fixed (12,12,12): the equilibrium the coil field was fitted to", fx, "#8a8984", "-"), ("DESC free (12,12,12)", dfree, BLUE[3], "-"),
                       ("VMEC++ free 13/12, ns 101", f13, ORNG[3], "--")], pl, "Flux surfaces, " + t)
fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=8.5)
fig.savefig("report/c08_free.png", dpi=130)


def vmec_spec(path):
    w = Dataset(path); w.set_auto_mask(False); xm = w["xm"][:]; a = np.hypot(w["rmnc"][-1], w["zmns"][-1])
    return np.arange(int(xm.max()) + 1), np.array([a[xm == m].max() for m in range(int(xm.max()) + 1)])


def desc_spec(path):
    import desc.io
    sf = desc.io.load(path).surface
    m = np.abs(np.concatenate([sf.R_basis.modes[:, 1], sf.Z_basis.modes[:, 1]])); a = np.abs(np.concatenate([sf.R_lmn, sf.Z_lmn]))
    return np.arange(m.max() + 1), np.array([a[m == k].max() for k in range(m.max() + 1)])


spec = [("fixed boundary", vmec_spec("c08/wout_vmec_fixed_m13_n12_ns101.nc"), "#8a8984", "-", "o"),
        ("DESC free (12,12,12)", desc_spec("c08/desc_free_L12_M12_N12.h5"), BLUE[3], "-", "o"),
        ("VMEC++ free 9/8, ns 101", vmec_spec("c08/wout_vmec_free_m9_n8_ns101.nc"), ORNG[1], "--", "s"),
        ("VMEC++ free 13/12, ns 101", vmec_spec("c08/wout_vmec_free_m13_n12_ns101.nc"), ORNG[3], "--", "s")]
if os.path.exists("c08/wout_vmec_free_m9_n8_ns201_vac64.nc"):
    spec.insert(3, ("VMEC++ free 9/8, ns 201, fine vacuum", vmec_spec("c08/wout_vmec_free_m9_n8_ns201_vac64.nc"), ORNG[2], "--", "^"))
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), constrained_layout=True)
for lab, (m, a), col, ls, mk in spec:
    ax[0].semilogy(m, np.maximum(a, 1e-9), color=col, ls=ls, marker=mk, ms=4, lw=1.4, label=lab)
ax[0].set_xlabel("poloidal mode number m"); ax[0].set_ylabel("max over n of |(R, Z)ₘₙ| [m]"); ax[0].set_ylim(1e-7, 20)
ax[0].set_title("Boundary Fourier spectrum", fontsize=10, loc="left", color=INK); ax[0].legend(fontsize=7.5, loc="upper right")
V = json.load(open("c08/vc_residuals.json"))
rows = [(k, lab) for k, lab in [("desc_fixed_L12_M12_N12_jacobi", "DESC fixed 12³"), ("desc_free_L12_M12_N12", "DESC free 12³"),
                                ("vmec_fixed_m13_n12_ns101", "VMEC++ fixed 13/12"), ("vmec_free_m9_n8_ns101", "VMEC++ free 9/8 ns101"),
                                ("vmec_free_m9_n8_ns101_vac64", "… fine vacuum"), ("vmec_free_m9_n8_ns201_vac64", "… ns 201, fine vac."),
                                ("vmec_free_m13_n12_ns101", "VMEC++ free 13/12")] if k in V]
x = np.arange(len(rows))
bn = [V[k]["48"]["rms_Bn"] for k, _ in rows]; pp = [V[k]["48"]["rms_P"] for k, _ in rows]
cols = [BLUE[3] if k.startswith("desc") else ORNG[3] for k, _ in rows]
ax[1].bar(x - 0.2, bn, 0.38, color=cols, alpha=0.55, label="B·n / |B|, rms")
ax[1].bar(x + 0.2, pp, 0.38, color=cols, label="pressure balance / B², rms")
ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels([l for _, l in rows], rotation=25, ha="right", fontsize=8)
ax[1].set_title("Free-boundary residual on each result's own LCFS field (virtual casing, same coil field)", fontsize=10, loc="left", color=INK)
ax[1].legend(fontsize=7.5, loc="upper left"); ax[1].grid(axis="x", visible=False)
fig.savefig("report/c08_free_diag.png", dpi=130)
print("figures written")
