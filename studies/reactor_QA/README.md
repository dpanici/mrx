# reactor_QA: DESC vs VMEC++ (fixed and free boundary) and MRX relaxation

This study compares DESC 0.17.3 and VMEC++ 0.9.3 on the reactor_QA DESC example, with its toroidal current made even in ρ. It uses resolution ladders in fixed boundary and a shared REGCOIL coil field in free boundary. MRX (this repo, at dd1942e) relaxes every fixed-boundary result to check whether its nested surfaces survive.

The write-up is `report/index.html`; its figures and `tables.json` are in `report/`.

## Layout

- `./`: the original even-current case.
  - `01_even_current.py` builds `reactor_QA_even.h5`.
  - `03_vmec.py` runs VMEC++, `05_desc_fixed.py` runs DESC fixed boundary, and `06_compare.py` compares results.
  - `08_mrx_run.sh` runs MRX (`scripts/relax.py` + `scripts/poincare_trace.py`) and `09_mrx_plot.py` plots it.
- `c08/`: the scaled case, with the current × 0.8. `10_make_case.py` writes `reactor_QA_c08.h5` and `input.reactor_QA_c08`. The scripts follow the same numbering, plus the free-boundary ones:
  - `02_coil_field.py`: REGCOIL fit, which writes `coil_field.h5` and `coil_field_grid.pkl`.
  - `03_vmec.py free ...`: VMEC++ free boundary through a Biot–Savart callback. Env: `NVACSKIP`, `NTHETA`, `NZETA`, `NSARR`, `FTOL`, `TAGSUF`.
  - `04_desc_free.py`: DESC free boundary (`proximal-lsq-exact` on `BoundaryError`). Run it with `DESC_MEM=6`; without that it runs out of memory on a 15 GB machine.
  - `17_free_compare.py`: compares the geometry of the two free-boundary results.
  - `21_bgrid_conv.py`: checks quadrature convergence of `BoundaryError`.
  - `22_vc_residual.py`: free-boundary residual of any result (DESC `.h5` or VMEC++ wout), computed from its own LCFS field with DESC's virtual-casing integral. Results go to `vc_residuals.json`.
  - `18`–`20`: diagnostics behind the report text.
    - `18`: VMECIO loading. Too inaccurate at the edge to use.
    - `19`: sign and |B| check on the shared boundary.
    - `20`: DESC solve on the VMEC++ free boundary. It did not converge (force error 0.5), so `desc_on_vmec_free_m13_n12_ns101.h5` is not a valid equilibrium.
- `scan/`: VMEC++ convergence test of the current scale factor.
- `13_report_figs.py`, `14_report_tables.py`, `15_report_html.py`: rebuild `report/`. Run them from this directory.

## Not included (regenerate with the scripts)

- VMEC++ `wout_*.nc`: 322 MB. Rerun `03_vmec.py`.
- MRX checkpoints and Poincaré traces (`mrx_*/checkpoints`, `trace.npz`): about 180 MB. Rerun `08_mrx_run.sh`. `report/tables.json` keeps the metrics computed from them.

The `*.pkl` files hold the evaluated surfaces, ι and scalars used by the comparisons. `*.log` are the solver logs.
