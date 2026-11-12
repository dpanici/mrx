"""HTML table fragments for the write-up, computed from the saved results (report/tables.json)."""
import json, os, glob, pickle, numpy as np, netCDF4
from common import compare, RHOS

def L(p): return pickle.load(open(p + ".pkl", "rb"))

def sqrtg_ratio(wout):
    th = np.linspace(0, 2 * np.pi, 128, endpoint=False); ph = np.linspace(0, np.pi, 64)
    d = netCDF4.Dataset(wout); g = np.asarray(d["gmnc"][:]); xm = np.asarray(d["xm_nyq"][:]); xn = np.asarray(d["xn_nyq"][:]); ns = g.shape[0]
    C = np.cos(xm[:, None, None] * th[None, :, None] - xn[:, None, None] * ph[None, None, :]).reshape(len(xm), -1)
    G = g[1:] @ C; sh = (np.arange(1, ns) - 0.5) / (ns - 1); ratio = np.abs(G).min(1) / np.abs(G).max(1)
    m = sh > 0.3; j = np.argmin(np.where(m, ratio, 9)); return float(ratio[j]), float(sh[j])

def label(tag):
    if tag.startswith("desc_fromvmec"):
        return "DESC (12,12,12) from VMEC++ start"
    if tag.startswith("desc"):
        p = tag.split("_"); lmn = "(%s,%s,%s)" % (p[2][1:], p[3][1:], p[4][1:]); return "DESC %s%s" % (lmn, " jacobi-dna" if "dna" in tag else "")
    p = tag.split("_"); return "VMEC++ %s/%s, ns %s" % (p[2][1:], p[3][1:], p[4][2:])

def ladder(prefix, ref, desc_tags, vmec_tags):
    R = L(prefix + ref); rows = []
    for t in vmec_tags + desc_tags:
        if not os.path.exists(prefix + t + ".pkl"):
            continue
        r = L(prefix + t); io = np.abs(r["iota"]); c = compare(r, R)
        row = dict(tag=t, label=label(t), code="vmec" if t.startswith("vmec") else "desc", ref=(t == ref),
                   iota_axis=float(np.interp(0.01, r["s"], io)), iota_max=float(io.max()), iota_edge=float(io[-1]),
                   d=[float(1e3 * x) for x in c["surf_dist"].max(0)[:-1]], axis=float(1e3 * c["axis"]), diota=float(c["iota"]),
                   wall=float(r.get("wall", np.nan)))
        if t.startswith("vmec"):
            row["fsq"] = float(r["fsq"]); row["sqrtg"], row["sqrtg_s"] = sqrtg_ratio(prefix + "wout_" + t + ".nc")
        else:
            row["force"] = float(r["force"])
        rows.append(row)
    return rows

def mrx_rows(prefix, pairs):
    rows = []
    for tag, pk, note in pairs:
        f = prefix + "mrx_" + tag
        if not os.path.exists(f + "/trace.npz"):
            rows.append(dict(tag=tag, label=label(pk) + note, missing=True)); continue
        tr = np.load(f + "/trace.npz"); js = json.load(open(f + "/relax.json"))
        r0 = tr["best_seed_r"]; d = np.zeros_like(r0)
        for z in (0, 0.25, 0.5):
            d = np.fmax(d, np.nanmax(np.abs(tr["best_zeta%g_logr" % z] - r0[:, None]), axis=1))
        keep = tr["best_keep"]; k = int(np.nanargmax(np.where(keep, d, -1)))
        rows.append(dict(tag=tag, label=label(pk) + note, res=js["params"]["resolution"], F0=js["trace"]["F"][0],
                         Fb=min(js["trace"]["F"]), dep=float(d[k]), at_r=float(r0[k]), at_iota=float(abs(tr["best_iota"][k])),
                         chaotic=int((keep & tr["best_chaotic"]).sum()), n=int(keep.size)))
    return rows

out = {}
out["orig"] = ladder("", "vmec_fixed_m17_n16_ns201",
                     ["desc_fixed_L8_M8_N8_jacobi", "desc_fixed_L8_M8_N8_jacobi-dna", "desc_fixed_L12_M12_N12_jacobi", "desc_fixed_L12_M12_N12_jacobi-dna"],
                     ["vmec_fixed_m9_n8_ns51", "vmec_fixed_m9_n8_ns101", "vmec_fixed_m13_n12_ns101", "vmec_fixed_m9_n8_ns201", "vmec_fixed_m9_n8_ns401", "vmec_fixed_m17_n16_ns201"])
out["c08"] = ladder("c08/", "vmec_fixed_m21_n20_ns201",
                    ["desc_fixed_L8_M8_N8_jacobi", "desc_fixed_L8_M8_N8_jacobi-dna", "desc_fixed_L12_M12_N12_jacobi", "desc_fixed_L12_M12_N12_jacobi-dna",
                     "desc_fromvmec_L12_M12_N12", "desc_fixed_L16_M16_N16_jacobi", "desc_fixed_L20_M12_N12_jacobi", "desc_fixed_L20_M16_N16_jacobi"],
                    ["vmec_fixed_m9_n8_ns51", "vmec_fixed_m9_n8_ns101", "vmec_fixed_m13_n12_ns101", "vmec_fixed_m9_n8_ns201", "vmec_fixed_m9_n8_ns401",
                     "vmec_fixed_m13_n12_ns201", "vmec_fixed_m17_n16_ns201", "vmec_fixed_m21_n20_ns201"])
D = lambda l, m, n, dna="": "desc_fixed_L%d_M%d_N%d_jacobi%s" % (l, m, n, dna)
out["mrx_orig"] = mrx_rows("", [("desc_L8_jacobi", D(8, 8, 8), ""), ("desc_L8_jacobi-dna", D(8, 8, 8, "-dna"), ""), ("desc_L12_jacobi", D(12, 12, 12), ""),
                                ("desc_L12_jacobi-dna", D(12, 12, 12, "-dna"), ""), ("vmec_m9_ns51", "vmec_fixed_m9_n8_ns51", ""),
                                ("vmec_m9_ns101", "vmec_fixed_m9_n8_ns101", ""), ("vmec_m13_ns101", "vmec_fixed_m13_n12_ns101", "")])
out["mrx_c08"] = mrx_rows("c08/", [("desc_L8_jacobi", D(8, 8, 8), ""), ("desc_L8_jacobi-dna", D(8, 8, 8, "-dna"), ""), ("desc_L12_jacobi", D(12, 12, 12), ""),
                                   ("desc_L12_jacobi-dna", D(12, 12, 12, "-dna"), ""), ("desc_fromvmec_L12", "desc_fromvmec_L12_M12_N12", ""),
                                   ("desc_L16_jacobi", D(16, 16, 16), ""), ("desc_L16_jacobi_wout_r32", D(16, 16, 16), " via wout"),
                                   ("vmec_m9_ns51", "vmec_fixed_m9_n8_ns51", ""), ("vmec_m9_ns101", "vmec_fixed_m9_n8_ns101", ""),
                                   ("vmec_m9_ns201", "vmec_fixed_m9_n8_ns201", ""), ("vmec_m9_ns201_r32", "vmec_fixed_m9_n8_ns201", ""),
                                   ("vmec_m9_ns401", "vmec_fixed_m9_n8_ns401", ""), ("vmec_m13_ns101", "vmec_fixed_m13_n12_ns101", ""),
                                   ("vmec_m13_ns201", "vmec_fixed_m13_n12_ns201", ""), ("vmec_m17_ns201", "vmec_fixed_m17_n16_ns201", ""),
                                   ("vmec_m21_ns201", "vmec_fixed_m21_n20_ns201", ""), ("vmec_m21_ns201_r32", "vmec_fixed_m21_n20_ns201", "")])
free = []
import os, re
VC = json.load(open("c08/vc_residuals.json"))
dfree = L("c08/desc_free_L12_M12_N12"); fixd = L("c08/desc_fixed_L12_M12_N12_jacobi")


def delbsq(tag):
    if not os.path.exists("c08/%s.log" % tag):
        return None
    rows = [l for l in open("c08/%s.log" % tag) if re.match(r"^ +[0-9]+ \|", l)]
    return float(rows[-1].split("|")[-1]) if rows else None


for tag, lab in [("desc_free_L12_M12_N12", "DESC free (12,12,12)"), ("vmec_free_m9_n8_ns51", "VMEC++ free 9/8, ns 51"),
                 ("vmec_free_m9_n8_ns101", "VMEC++ free 9/8, ns 101"), ("vmec_free_m9_n8_ns101_vac64", "VMEC++ free 9/8, ns 101, fine vacuum"),
                 ("vmec_free_m9_n8_ns201_vac64", "VMEC++ free 9/8, ns 201, fine vacuum"), ("vmec_free_m9_n8_ns401_vac64", "VMEC++ free 9/8, ns 401, fine vacuum"), ("vmec_free_m13_n12_ns101", "VMEC++ free 13/12, ns 101"),
                 ("vmec_free_m13_n12_ns201_vac64", "VMEC++ free 13/12, ns 201, fine vacuum (crinkled, see below)")]:
    if not os.path.exists("c08/%s.pkl" % tag):
        continue
    r = L("c08/" + tag); code = "desc" if tag.startswith("desc") else "vmec"
    c = compare(r, fixd); cd = compare(r, dfree); io = np.abs(r["iota"]); v = VC.get(tag, {}).get("48", {})
    free.append(dict(label=lab, code=code, lcfs=float(1e3 * c["lcfs"]), lcfs_desc=None if code == "desc" else float(1e3 * cd["lcfs"]),
                     d_desc=None if code == "desc" else [float(1e3 * x) for x in cd["surf_dist"].max(0)[:-1]],
                     axis_desc=None if code == "desc" else float(1e3 * cd["axis"]), diota_desc=None if code == "desc" else float(cd["iota"]),
                     rms_Bn=v.get("rms_Bn"), rms_P=v.get("rms_P"), delbsq=delbsq(tag), V=float(r["V"]), beta=float(r["beta"]),
                     iota_edge=float(io[-1]), wall=float(r.get("wall", np.nan))))
free_ref = {k: VC[k]["48"] for k in ["desc_fixed_L12_M12_N12_jacobi", "vmec_fixed_m13_n12_ns101"] if k in VC}
out["free_ref"] = free_ref
out["free"] = free
json.dump(out, open("report/tables.json", "w"), indent=1)
for k in out: print(k, len(out[k]), "rows")
