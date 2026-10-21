"""Self-consistent inputs for the paper: chi(F) table for the triple well, and state sets
(energies, dipoles, depolarisation-shifted transitions, equilibrium populations, N) for the triple
well (at its optimum field) and the ADQW (0 V) at several sheet densities. Writes tables.json."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import sp
HW = 1.23984 / 10
TW = [(58, "well"), (14, "barrier"), (34, "well"), (14, "barrier"), (58, "well")]
AD = [(64, "well"), (10, "barrier"), (29, "well")]
NS = [1e10, 3e10, 1e11, 2e11, 5e11]

def pack(o):
    return dict(E=o["E"].tolist(), Z=o["Z"].tolist(), Et=o["Et"].tolist(), rho=o["rho"].tolist(), N=o["N"],
                chi=[o["chi"].real, o["chi"].imag])

def chiF(_):
    rows, VH = [], None
    for F in range(0, 115, 5):
        o = sp.chi2_sc(TW, float(F), HW, dz=0.5, VH0=VH); VH = o["r"]["VH"]
        rows.append(dict(F=F, re=o["chi"].real, im=o["chi"].imag, E=(o["E"] - o["E"][0]).tolist()))
    return rows

def states(ns):
    best, VH = None, None
    for F in range(35, 80, 5):
        o = sp.chi2_sc(TW, float(F), HW, ns_cm2=ns, dz=0.5, VH0=VH); VH = o["r"]["VH"]
        if best is None or abs(o["chi"]) > abs(best[1]["chi"]): best = (F, o)
    a = sp.chi2_sc(AD, 0.0, HW, ns_cm2=ns, dz=0.5)
    return dict(ns=ns, F_tw=best[0], tw=pack(best[1]), ad=pack(a))

if __name__ == "__main__":
    with Pool(2) as p:
        res = p.map(states, NS)
    out = dict(chiF=chiF(0), states=res)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables.json"), "w"))
    for r in res:
        print(f"ns {r['ns']:.0e}: TW best F {r['F_tw']} |chi| {np.hypot(*r['tw']['chi'])/1e3:.2f} nm/V ; ADQW {np.hypot(*r['ad']['chi'])/1e3:.2f} nm/V ; ratio {np.hypot(*r['tw']['chi'])/np.hypot(*r['ad']['chi']):.3f}")
    print("chiF:", [(r['F'], round(np.hypot(r['re'], r['im'])/1e3, 1)) for r in out['chiF']])
