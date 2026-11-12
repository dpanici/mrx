"""MRX smoke test: read a DESC .h5 and a VMEC++ wout, build the sequence, the initial field and its diagnostics."""
import sys, time
from mrx.equilibria import read_equilibrium
from mrx.geometry import build_sequence
from mrx.nullspace import compute_nullspaces
from mrx.relaxation.initial_conditions import initial_field

res = tuple(int(x) for x in sys.argv[1:4])
for path in sys.argv[4:]:
    t = time.time()
    st = read_equilibrium(path)
    print(path, st["kind"], "nfp", st["nfp"], "theta_reversed", st.get("theta_reversed"), flush=True)
    seq = build_sequence(path, res, 2)
    compute_nullspaces(seq)
    B0, ic = initial_field(seq)
    print("  %.0fs  " % (time.time() - t) + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in ic.items()), flush=True)
