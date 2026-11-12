"""The c08 case: reactor_QA_even (even current, bootstrap-consistent) with the current profile multiplied by 0.8,
so that the reversed-shear iota peak (~0.61) stays clear of the 2/3 resonance. Fixed-boundary DESC solve at
(8, 8, 8) from the reactor_QA_even equilibrium, then the VMEC++ input written from it by DESC's VMECIO."""
import numpy as np
import desc.io
from desc.profiles import PowerSeriesProfile
from desc.vmec import VMECIO

SCALE = 0.8
eq = desc.io.load("../reactor_QA_even.h5")
c = eq.current
eq.current = PowerSeriesProfile(SCALE * np.asarray(c.params), c.basis.modes[:, 0], sym="even")
print("I(1): %.4e -> %.4e" % (float(c(1.0)[0]), float(eq.current(1.0)[0])))
eq, _ = eq.solve(verbose=2, copy=True, ftol=1e-10, xtol=1e-12, gtol=1e-12, maxiter=150)
eq.save("reactor_QA_c08.h5")
VMECIO.write_vmec_input(eq, "input.reactor_QA_c08")
