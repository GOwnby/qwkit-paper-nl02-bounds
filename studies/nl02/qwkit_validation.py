"""Validation of the three-level theory against full qwkit simulations of the N1 structures
(self-consistent multilevel states, depolarisation-shifted transitions, LO-phonon rate matrices, TM waveguide).
A. The absorption-nonlinearity identity |chi|^2 = Q alpha1 alpha2 vs pump photon energy.
B. Saturated, phase-matched propagation eta(P) vs the mapped universal ladder model, Theorem 1 and the
   modal form of Theorem 2.  Writes qwkit_validation.json."""
import sys, os, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", ".."))
sys.path.insert(0, os.path.join(here, "..", "n1_paper"))
sys.path.insert(0, here)
import numpy as np
from multiprocessing import Pool
from qwkit import propagation as pr
import propagation_study as ps
from model3 import Ladder, g_of_u

QE, EPS0, C0, HBAR = pr.QE, pr.EPS0, pr.C0, pr.HBAR
GAM = 0.010; NREF = 3.3; HW = ps.HW
RATES = json.load(open(os.path.join(here, "..", "n1_paper", "rates.json")))


def R_for(i, d):
    return np.array([r for r in RATES if r["i"] == i and r["design"] == d][0]["R"])


def ladder_idx(st, hw):
    Z = np.array(st["Z"]); Et = np.array(st["Et"]); n = len(Z)
    lor = lambda d: 1 / (1 + (d / GAM) ** 2)
    b = max(range(1, n), key=lambda m: Z[0, m] ** 2 * lor(Et[m, 0] - hw))
    c = max([m for m in range(1, n) if m != b], key=lambda m: abs(Z[0, b] * Z[b, m] * Z[m, 0]) * lor(Et[m, 0] - 2 * hw))
    return 0, b, c


def Q3(z23_A, hw):
    w = hw * QE / HBAR
    return QE ** 2 * (z23_A * 1e-10) ** 2 * NREF ** 2 * C0 ** 2 / (2 * w ** 2 * (GAM * QE) ** 2)


def identity_scan(st, hws):
    a, b, c = ladder_idx(st, HW)
    Z = np.array(st["Z"])
    out = []
    for hw in hws:
        med = pr.Medium(st, hw, gamma=GAM, nref=NREF)
        rho = med.rho0
        chi = med.chi(rho); a1, a2 = med.alpha(rho)
        out.append(dict(hw=float(hw), chi=float(abs(chi)), a1=float(a1), a2=float(a2),
                        ratio=float(abs(chi) ** 2 / (a1 * a2) / Q3(Z[b, c], hw))))
    return dict(ladder=[a, b, c], z12=float(Z[a, b]), z13=float(Z[a, c]), z23=float(Z[b, c]), scan=out)


def mapped_ladder(st, R, w):
    """Map a multilevel waveguide medium onto the universal ladder (Sec. 7 of the paper)."""
    a, b, c = ladder_idx(st, HW)
    Z = np.array(st["Z"]); Et = np.array(st["Et"]); N = st["N"] * 1e6
    om = HW * QE / HBAR
    z12, z13, z23 = abs(Z[a, b]) * 1e-10, abs(Z[a, c]) * 1e-10, abs(Z[b, c]) * 1e-10
    s12 = om * QE ** 2 * z12 ** 2 / (NREF * C0 * EPS0 * GAM * QE)            # resonant cross-sections (m^2)
    s13 = 2 * om * QE ** 2 * z13 ** 2 / (NREF * C0 * EPS0 * GAM * QE)
    A12 = N * s12
    E = np.array(st["E"])
    k2 = sum(R[b, m] for m in range(len(E)) if E[m] < E[b])                   # total decay out of level 2
    k3_2 = R[c, b]; k3_1 = sum(R[c, m] for m in range(len(E)) if E[m] < E[c] and m != b)
    tau21 = 1 / k2
    Is = HBAR * om / (2 * s12 * tau21)                                          # W/m^2 (stack-averaged)
    area = w["t_stack_um"] * 1e-6 * w["w_um"] * 1e-6
    Ps = Is * area / w["G1"]
    O = w["O_um"] * 1e6 / (w["w_um"] * 1e-6)
    K = (np.pi / (w["lam1_um"] * 1e-6)) * np.sqrt(2 * O / (w["n1"] ** 2 * w["n2"] * C0 * EPS0))
    chiref = N * QE ** 3 * z12 * z23 * z13 / (EPS0 * (GAM * QE) ** 2)
    cc = K * chiref * np.sqrt(Ps) / (w["G1"] * A12)
    G1, G2 = w["G1"], w["G2"]
    pars = dict(r23=(z23 / z12) ** 2, u0=(G1 / G2) * (z12 / z13) ** 2,
                d12=(Et[b, a] - HW) / GAM, d13=(Et[c, a] - 2 * HW) / GAM,
                t32=k3_2 / k2, t31=k3_1 / k2,
                b1=w["abg1_cm"] * 100 / (G1 * A12), b2=w["abg2_cm"] * 100 / (G1 * A12))
    zm = max(z12, z23)
    Qm = QE ** 2 * zm ** 2 * NREF ** 2 * C0 ** 2 / (2 * om ** 2 * (GAM * QE) ** 2)
    bound_per_W = K ** 2 * Qm / (2 * G1 * G2)                                  # Theorem 2, modal form (1/W)
    return dict(pars=pars, c=float(cc), Ps_W=float(Ps), Is_MWcm2=float(Is / 1e10), tau21_ps=float(tau21 * 1e12),
                A12_cm=float(A12 / 100), bound_per_W=float(bound_per_W), K=float(K), ladder=[a, b, c])


def job(args):
    i, d, P = args
    st = ps.T["states"][i][d]
    R = R_for(i, d)
    med = pr.Medium(st, HW, gamma=GAM, R=R)
    w = ps.wg_for(st)
    Lmax = 3000.0 if P < 30 else 8000.0
    r = pr.propagate(med, w, P, "perfect", L_max_um=Lmax, steps_per_lcoh=16)
    return dict(i=i, design=d, ns=ps.T["states"][i]["ns"], P=float(P), eta=float(r["eta"]),
                L_opt_um=float(r["L_opt_um"]), at_end=bool(r["L_opt_um"] > 0.98 * Lmax))


if __name__ == "__main__":
    out = dict(identity={}, maps={}, prop=[], model={})
    hws = np.linspace(HW - 0.04, HW + 0.04, 33)
    for i in [2, 4]:
        for d in ["ad", "tw"]:
            key = f"{d}_{ps.T['states'][i]['ns']:.0e}"
            st = ps.T["states"][i][d]
            out["identity"][key] = identity_scan(st, hws)
            out["maps"][key] = mapped_ladder(st, R_for(i, d), ps.wg_for(st))
            print(key, out["maps"][key]["pars"], "Ps", out["maps"][key]["Ps_W"], "W  Is", out["maps"][key]["Is_MWcm2"],
                  "MW/cm2", flush=True)
    Ps_list = list(np.logspace(-2, np.log10(300), 22))
    jobs = [(i, d, P) for i in [2, 4] for d in ["ad", "tw"] for P in Ps_list]
    with Pool(4) as pool:
        out["prop"] = pool.map(job, jobs, chunksize=2)
    for key, m in out["maps"].items():
        lad = Ladder(**m["pars"], c=m["c"])
        res = []
        for P in Ps_list:
            p = P / m["Ps_W"]
            r = lad.propagate(p, npts=6000)
            res.append(dict(P=float(P), eta=float(r["eta"]), eta_ss=float(lad.eta_small_signal(p) * (m["c"] ** 2 * lad.u0 / lad.eta_s)),
                            bound=float(m["bound_per_W"] * P)))
        out["model"][key] = res
    json.dump(out, open(os.path.join(here, "qwkit_validation.json"), "w"), indent=1)
    for key in out["maps"]:
        d, ns = key.split("_")
        sim = [r for r in out["prop"] if r["design"] == d and f"{r['ns']:.0e}" == ns]
        mod = out["model"][key]
        print(key)
        for s, mm in zip(sim, mod):
            print(f"  P={s['P']:8.3f} W  qwkit {s['eta']*100:8.4f}%  ladder {mm['eta']*100:8.4f}%  ss {mm['eta_ss']*100:8.4f}%  "
                  f"bound {mm['bound']*100:8.3f}%  L={s['L_opt_um']:.0f}{' END' if s['at_end'] else ''}")
