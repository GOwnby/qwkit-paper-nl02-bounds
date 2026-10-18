"""Length-limited bound on the generated harmonic (generalizes Almogy, PhD thesis, Caltech 1995, Eq. 9.37).

The harmonic is driven by the two-photon coherence rho_31. Writing rho_31 = G(e1) + (linear response to e2), the linear
part only absorbs (no inversion between 1 and 3), so |de2/dx| <= c2 |G(e1)| and, for any pump history, length,
detuning, phase-matching pattern |s(x)| <= 1 and background loss,
    |e2(X)| <= c2 S X,   S = sup_p |G(sqrt p)|,   c2 = 2 sqrt(pcoh/u0)          (x in units of 1/A12)
i.e. eta(X) <= (c2 S X)^2 / p0. Unlike Theorem 2 this bound does not depend on the pump intensity, only on the
(resonant) optical depth. G and S come from the full steady-state density matrix (dm_check.DM); the rate-equation
ladder gives the same S to within its validity range. For any density matrix |rho_31| <= 1/2, so S <= 1/2 always.

Checks: (i) the symmetric case of Almogy's Eq. 9.37a (equal dipoles, double resonance, all excited levels decaying at
1/T1, all coherences at 1/T2); (ii) every density-matrix propagation of dm_check.json obeys the bound; (iii) the
minimum length the bound implies for the efficiencies actually reached. Writes length_bound.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import numpy as np
from scipy.optimize import minimize_scalar
from dm_check import DM
from study_universal import CASES

PGRID = np.logspace(-3, 6, 400)


def S_sup(dm):
    g = np.array([abs(dm.steady(np.sqrt(p) + 0j, 0.0)[2, 0]) for p in PGRID])
    i = int(np.argmax(g))
    lo, hi = np.log(PGRID[max(i - 1, 0)]), np.log(PGRID[min(i + 1, len(PGRID) - 1)])
    r = minimize_scalar(lambda lp: -abs(dm.steady(np.exp(lp / 2) + 0j, 0.0)[2, 0]), bounds=(lo, hi), method="bounded")
    return float(-r.fun), float(np.exp(r.x))


if __name__ == "__main__":
    out = dict(symmetric=[], cases={}, dm_check=[])
    # (i) Almogy Eq. 9.37a,b: I_2w/I_2w^sat <= [z/(sqrt(T2/T1) + 3 sqrt(T1/T2))]^2, z = A12 L. For equal dipoles and
    # indices (u0 = 1) I_2w^sat equals our Is, so the comparison is (c2 S)^2 against 1/(sqrt(1/pcoh) + 3 sqrt(pcoh))^2.
    for pcoh in (1.0, 5.0, 15.0, 50.0):
        dm = DM(r23=1.0, u0=1.0, t32=0.0, t31=1.0, pcoh=pcoh)
        S, pS = S_sup(dm)
        ours = (2 * np.sqrt(pcoh) * S) ** 2
        ay = 1 / (np.sqrt(1 / pcoh) + 3 * np.sqrt(pcoh)) ** 2
        out["symmetric"].append(dict(T1_T2=pcoh, S=S, p_at_S=pS, ours_I2_per_X2=float(ours), almogy_I2_per_X2=float(ay)))
        print(f"symmetric T1/T2={pcoh:4.0f}: S={S:.5f} (Almogy 1/[2(1+3T1/T2)] = {1/(2*(1+3*pcoh)):.5f}); "
              f"I2/(Is X^2) bound ours {ours:.6f} vs Almogy Eq.9.37 {ay:.6f}", flush=True)
    for name in ("balanced", "ADQW-like", "TW-like", "weak 2-3"):
        pars = dict(CASES[name]); pars["pcoh"] = 15.0
        S, pS = S_sup(DM(**pars))
        out["cases"][name] = dict(S=S, p_at_S=pS, c2=float(2 * np.sqrt(15.0 / pars["u0"])))
        print(f"{name:10s}: S = {S:.4f} at p = {pS:.3g} (|rho31| <= 1/2 always)", flush=True)
    for r in json.load(open(os.path.join(here, "dm_check.json"))):
        pars = r["pars"]; name = r["tag"].split("|")[1] if r["tag"].startswith("res") else "ADQW-like"
        S = out["cases"][name]["S"] if r["tag"].startswith("res") else S_sup(DM(**pars))[0]
        c2 = 2 * np.sqrt(pars["pcoh"] / pars["u0"])
        bound = (c2 * S * r["L_dm"]) ** 2 / r["p0"]
        Lmin = np.sqrt(r["eta_dm"] * r["p0"]) / (c2 * S)
        out["dm_check"].append(dict(tag=r["tag"], p0=r["p0"], eta=r["eta_dm"], L=r["L_dm"], bound_at_L=float(bound), ratio=float(r["eta_dm"] / bound),
                                    Lmin=float(Lmin), L_over_Lmin=float(r["L_dm"] / Lmin)))
    rat = [d["ratio"] for d in out["dm_check"]]; lr = [d["L_over_Lmin"] for d in out["dm_check"]]
    print(f"dm_check propagations: eta/bound max {max(rat):.3f}, median {np.median(rat):.3f}; L_opt/L_min range {min(lr):.2f}-{max(lr):.2f}", flush=True)
    json.dump(out, open(os.path.join(here, "length_bound.json"), "w"), indent=1)
