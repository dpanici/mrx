"""Write a DESC equilibrium as a VMEC wout (DESC's VMECIO.save) so MRX can read it as radial splines, and measure
the conversion error of the surfaces: python 16_desc_to_wout.py DESC.h5 WOUT.nc"""
import sys, pickle, numpy as np, desc.io, vmecpp
from desc.vmec import VMECIO
from common import vmec_eval, compare
eq = desc.io.load(sys.argv[1])
VMECIO.save(eq, sys.argv[2], surfs=201, verbose=1)
r = vmec_eval(vmecpp.VmecWOut.from_wout_file(sys.argv[2]))
c = compare(r, pickle.load(open(sys.argv[1].replace(".h5", ".pkl"), "rb")))
print("wout vs DESC surfaces: rho .2-.8 [mm]", np.round(1e3 * c["surf_dist"].max(0)[:-1], 2), "axis %.2f mm, iota %.1e" % (1e3 * c["axis"], c["iota"]))
