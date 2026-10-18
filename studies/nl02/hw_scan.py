"""Full qwkit check of detuned pumping for the ADQW and TW (ns = 5e11): pump photon energy scanned at
fixed pump power (detunes d12 and d13 together, d13 = 2 d12). The waveguide (TM modes at lambda and lambda/2,
confinement, overlap, indices, background loss including free-carrier and gold dispersion) is recomputed at
every pump wavelength with the N1 layer stack. Writes hw_scan.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..")); sys.path.insert(0, os.path.join(here, "..", "n1_paper")); sys.path.insert(0, here)
import numpy as np
from multiprocessing import Pool
from qwkit import propagation as pr
import propagation_study as ps
import waveguide_design as wd
from qwkit import waveguide as wgm
from qwkit_validation import R_for, GAM

def wg_at(lam1):
    """N1 waveguide (t_up 2.0, t_sp 0.9 um) recomputed at pump wavelength lam1 (um)."""
    L = wd.stack_layers(2.0, 0.9)
    m1 = wgm.tm_modes(L, lam1, 0.01, 1)[0]; m2 = wgm.tm_modes(L, lam1 / 2, 0.01, 1)[0]
    z0 = 0.1 + 2.0 + 0.9; z1 = z0 + wd.T_STACK
    return dict(n1=float(m1["neff"].real), n2=float(m2["neff"].real), G1=float(wgm.overlap(m1, z0, z1)),
                G2=float(wgm.overlap(m2, z0, z1)), O_um=float(wgm.shg_overlap(m1, m2, z0, z1)),
                abg1_cm=float(m1["alpha_cm"]), abg2_cm=float(m2["alpha_cm"]), t_stack_um=40 * 378e-4, w_um=15.0,
                Lcoh_um=20.0, lam1_um=float(lam1))

def job(args):
    d, P, dhw, w = args
    st = ps.T["states"][4][d]; R = R_for(4, d)
    hw = ps.HW + dhw
    med = pr.Medium(st, hw, gamma=GAM, R=R)
    r = pr.propagate(med, w, P, "perfect", L_max_um=6000.0, steps_per_lcoh=16)
    return dict(design=d, P=float(P), dhw=float(dhw), eta=float(r["eta"]), L_opt_um=float(r["L_opt_um"]))

if __name__ == "__main__":
    xs = [float(x) for x in np.linspace(-0.06, 0.015, 16)]
    with Pool(2) as pool:
        WGS = dict(zip(xs, pool.map(wg_at, [1.23984 / (ps.HW + x) for x in xs])))
    for x in xs:
        print("dhw %+.0f meV lam %.2f um: G1 %.3f G2 %.3f O %.4f n1 %.4f n2 %.4f loss %.2f/%.2f" % (x * 1e3, WGS[x]["lam1_um"],
              WGS[x]["G1"], WGS[x]["G2"], WGS[x]["O_um"], WGS[x]["n1"], WGS[x]["n2"], WGS[x]["abg1_cm"], WGS[x]["abg2_cm"]), flush=True)
    jobs = [(d, P, x, WGS[x]) for d in ["ad", "tw"] for P in [0.01, 1.0, 4.0, 10.0, 30.0] for x in xs]
    with Pool(2) as pool:
        res = pool.map(job, jobs, chunksize=2)
    json.dump(res, open(os.path.join(here, "hw_scan.json"), "w"))
    json.dump({f"{k:.4f}": v for k, v in WGS.items()}, open(os.path.join(here, "hw_scan_wg.json"), "w"))
    for d in ["ad", "tw"]:
        for P in [0.01, 1.0, 4.0, 10.0, 30.0]:
            rr = [r for r in res if r["design"] == d and r["P"] == P]
            r0 = min(rr, key=lambda r: abs(r["dhw"])); b = max(rr, key=lambda r: r["eta"])
            print(d, P, "res %.4f%%" % (r0["eta"] * 100), "best %.4f%% at dhw=%+.0f meV L=%.0f" % (b["eta"] * 100, b["dhw"] * 1e3, b["L_opt_um"]))
