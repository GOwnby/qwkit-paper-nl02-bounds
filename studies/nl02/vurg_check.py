"""Cross-check of Vurgaftman, Meyer & Ram-Mohan, IEEE JQE 32, 1334 (1996), Table I structures with qwkit
(one-band, nonparabolic, orthonormalised states), a Thomas-Reiche-Kuhn cap on z23, and the Rabi energy of the
2-3 transition at their waveguide operating point. Writes vurg_check.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np
from qwkit import core
HB2M0 = 3.80998
rows = [("SI", [(26, "well"), (19.5, "barrier"), (57, "well")], dict(E12=140, E13=250, r13=1.13, r23=2.07)),
        ("WG", [(64.5, "well"), (18, "barrier"), (36.5, "well")], dict(E12=87, E13=219, r13=1.50, r23=3.20)),
        ("WG(PM)", [(64.5, "well"), (17, "barrier"), (36.5, "well")], dict(E12=88, E13=218, r13=1.27, r23=3.18)),
        ("WG(DR)", [(62, "well"), (20, "barrier"), (29, "well")], dict(E12=117, E13=234, r13=1.40, r23=2.21))]
out = []
for name, L, ref in rows:
    r = core.analyze(L, 0.0, 0.117, nstates=4)
    E, Z = r["E"], r["Z"]; E21 = E[1] - E[0]; E32 = E[2] - E[1]; z12 = abs(Z[0, 1])
    meff = 0.043 * (1 + E[1] / 0.75)
    zmax = np.sqrt((HB2M0 / meff + E21 * z12 ** 2) / E32)
    z12max_for_ref = np.sqrt(HB2M0 / meff / E32) / np.sqrt(ref["r23"] ** 2 - E21 / E32)
    out.append(dict(name=name, E21_meV=1e3 * E21, E31_meV=1e3 * (E[2] - E[0]), z12_nm=z12 / 10, z13_nm=abs(Z[0, 2]) / 10,
                    z23_nm=abs(Z[1, 2]) / 10, r13=abs(Z[0, 2] / Z[0, 1]), r23=abs(Z[1, 2] / Z[0, 1]), ref=ref,
                    r23_trk_max=zmax / z12, z12_max_nm_for_ref_ratio=z12max_for_ref / 10))
    print(out[-1])
op = []
for z12nm in [0.8, 1.2, 1.5]:
    for I in [20e10, 40e10]:
        E0 = np.sqrt(2 * I / (3.3 * 8.854e-12 * 2.998e8))
        op.append(dict(z12_nm=z12nm, I_MWcm2=I / 1e10, hOm23_meV=E0 * 3.2 * z12nm * 1e-9 * 1e3, D23_meV=float(np.hypot(15, 5))))
print(op)
json.dump(dict(structures=out, operating_point=op), open(os.path.join(here, "vurg_check.json"), "w"), indent=1)
