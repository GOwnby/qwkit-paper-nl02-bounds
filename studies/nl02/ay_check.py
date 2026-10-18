"""Independent check against Almogy & Yariv, LEOS'94 (doi 10.1109/LEOS.1994.586950), Eq. (2): for double
resonance, equal dipoles, all excited levels decaying to the ground state at 1/T1 and equal T2,
   chi_eff / chi_res = 1 / [1 + 2 (Omega T2)^2 (1 + 3 T1/T2)],   Omega = mu E0 / (2 hbar),
and their statement that SHG vanishes asymptotically for mu12/mu23 = sqrt(2). Compares the full density
matrix (dm_check.DM) and the rate-equation ladder (model3.Ladder). Writes ay_check.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import numpy as np
from dm_check import DM
from model3 import Ladder

out = dict(rows=[], zero={})
for pcoh in [5.0, 15.0, 50.0]:                       # T1/T2
    dm = DM(r23=1.0, u0=1.0, t32=0.0, t31=1.0, pcoh=pcoh)
    lad = Ladder(r23=1.0, u0=1.0, t32=0.0, t31=1.0, eta_s=1e-6)
    r0 = dm.steady(1e-4, 0)[2, 0] / 1e-8
    for OmT2 in [0.03, 0.1, 0.3, 1.0, 3.0]:
        e1 = 2 * OmT2 * np.sqrt(pcoh)                  # Omega_R T2 = |e1|/sqrt(pcoh), Omega = Omega_R/2
        dmv = abs(dm.steady(e1, 0)[2, 0] / e1 ** 2 / r0)
        rev = abs(lad.coeffs(lad.rho(e1 ** 2))[2] / lad.S0)
        ay = 1 / (1 + 2 * OmT2 ** 2 * (1 + 3 * pcoh))
        out["rows"].append(dict(T1_T2=pcoh, OmT2=OmT2, dm=dmv, rate_eq=rev, ay=ay))
        print(f"T1/T2={pcoh:4.0f} OmT2={OmT2:4.2f}  DM {dmv:.5f}  AY {ay:.5f}  rate-eq {rev:.5f}  (rate-eq form 1/(1+6 Om^2 T1T2) = {1/(1+6*OmT2**2*pcoh):.5f})")
# asymptotic zero: C_inf = 0 at r23 = (t32+t31)/(1+t31); decay to ground -> r23 = 1/2 -> mu12/mu23 = sqrt 2
r23 = (0.0 + 1.0) / (1 + 1.0)
out["zero"] = dict(mu12_over_mu23=float(1 / np.sqrt(r23)))
dmz = DM(r23=r23, u0=1.0, t32=0.0, t31=1.0, pcoh=15.0); dmo = DM(r23=1.0, u0=1.0, t32=0.0, t31=1.0, pcoh=15.0)
for p in [10.0, 100.0, 1000.0]:
    e1 = np.sqrt(p)
    out["zero"][f"p{p:g}"] = dict(chiI_zero=float(abs(dmz.steady(e1, 0)[2, 0])), chiI_equal=float(abs(dmo.steady(e1, 0)[2, 0])))
print(out["zero"])
json.dump(out, open(os.path.join(here, "ay_check.json"), "w"), indent=1)
