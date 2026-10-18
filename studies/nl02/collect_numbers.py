"""Collect the numbers quoted in the NL-02 manuscript into numbers.json."""
import os, json, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
V = json.load(open(os.path.join(here, "qwkit_validation.json")))
U = json.load(open(os.path.join(here, "universal.json")))
D = json.load(open(os.path.join(here, "detuning.json")))
B = json.load(open(os.path.join(here, "bottleneck.json")))
from model3 import g_of_u
HBAR_meVps = 0.6582119569; GAM = 10.0
out = {}
for key, m in V["maps"].items():
    p = m["pars"]; tau = m["tau21_ps"]
    eta_s = p["r23"] * HBAR_meVps / (4 * tau * GAM)
    pcoh = GAM * tau / HBAR_meVps
    d23 = p["d13"] - p["d12"]
    pval = pcoh * min(1 + p["d12"] ** 2, (1 + d23 ** 2) / p["r23"])
    Pval = pval * m["Ps_W"]
    d, ns = key.split("_")
    sim = sorted([r for r in V["prop"] if r["design"] == d and f"{r['ns']:.0e}" == ns], key=lambda r: r["P"])
    Ps = np.array([r["P"] for r in sim]); et = np.array([r["eta"] for r in sim])
    eta_val = float(np.exp(np.interp(np.log(Pval), np.log(Ps), np.log(et))))
    mod = V["model"][key]; etm = np.array([r["eta"] for r in mod])
    dev = np.abs(etm / et - 1)
    z = V["identity"][key]
    rat = [s["ratio"] for s in z["scan"]]; chis = [s["chi"] for s in z["scan"]]; a1 = [s["a1"] for s in z["scan"]]
    book = HBAR_meVps / (4 * tau * GAM) * (abs(z["z13"] * z["z23"]) / z["z12"] ** 2) ** 2
    out[key] = dict(pars=p, z12=z["z12"], z13=z["z13"], z23=z["z23"], tau21_ps=tau, Is_MWcm2=m["Is_MWcm2"], Ps_W=m["Ps_W"],
                    A12_modal_cm=m["A12_cm"] * 0, eta_s=eta_s, pcoh=pcoh, p_valid=pval, P_valid_W=Pval,
                    eta_qwkit_at_Pvalid=eta_val, bound_at_Pvalid=m["bound_per_W"] * Pval,
                    g_u0=float(g_of_u(p["u0"])), lowP_ratio_to_bound=float(et[0] / (m["bound_per_W"] * Ps[0])),
                    ladder_dev_max_Ple_Pvalid=float(max(dev[Ps <= Pval * 1.0001])) if any(Ps <= Pval) else None,
                    ladder_dev_max_all=float(max(dev)),
                    identity_ratio_range=[float(min(rat)), float(max(rat))],
                    chi_range=float(max(chis) / min(chis)), a1_range=float(max(a1) / min(a1)),
                    book_estimate=book, eta_qwkit_10W=float(np.exp(np.interp(np.log(10), np.log(Ps), np.log(et)))),
                    eta_qwkit_300W=float(et[-1]), Lopt_300W_um=float(sim[-1]["L_opt_um"]))
t2 = U["theorem2"]
out["theorem2"] = dict(n=len(t2), max_ratio=max(x["ratio"] for x in t2), median=float(np.median([x["ratio"] for x in t2])),
                       n_qpm=sum(1 for x in t2 if x["mode"] != "pm"), max_ratio_qpm=max(x["ratio"] for x in t2 if x["mode"] != "pm"),
                       inversions=sum(x["inversion"] for x in t2))
out["theorem1"] = U["theorem1"]
out["sat"] = {s["name"]: dict(s_inf=s["s_inf"], nu=s["nu"], C_inf=s["C_inf"], k_inf=s["k_inf"],
                              peak_le20=max(e for e, pp in zip(s["eta"], s["p"]) if pp <= 20),
                              p_peak=[pp for e, pp in zip(s["eta"], s["p"]) if pp <= 20][int(np.argmax([e for e, pp in zip(s["eta"], s["p"]) if pp <= 20]))],
                              film_peak=max(s["film"]), p_film_peak=s["p"][int(np.argmax(s["film"]))]) for s in U["sat"]}
det = {}
for n in ["ADQW-like", "TW-like"]:
    for p in [1.0, 10.0, 100.0]:
        for b in [1e-4, 1e-2]:
            rr = [r for r in D if r["name"] == n and r["p"] == p and r["b"] == b]
            best = max(rr, key=lambda r: r["x"]); r0 = min(rr, key=lambda r: abs(r["d12"]))
            det[f"{n}|{p:g}|{b:g}"] = dict(x0=r0["x"], best=best["x"], d12=best["d12"], d13=best["d13"], L=best["L"], gain=best["x"] / r0["x"])
out["detuning"] = det
out["bottleneck"] = {f"{r['f']:g}": r for r in B if abs(np.log10(r["P"]) - 1) < 0.05}
json.dump(out, open(os.path.join(here, "numbers.json"), "w"), indent=1)
for k, v in out.items():
    print(k, json.dumps(v, default=float)[:900])

# rate-equation validity limit for the detuned operating points of hw_scan (ladder levels of each structure)
import sys
sys.path.insert(0, os.path.join(here, "..", "..")); sys.path.insert(0, os.path.join(here, "..", "n1_paper"))
import propagation_study as ps
from qwkit_validation import ladder_idx
Hs = json.load(open(os.path.join(here, "hw_scan.json")))
val = {}
for d in ["ad", "tw"]:
    key = f"{d}_5e+11"; st = ps.T["states"][4][d]; Et = np.array(st["Et"]); a, b, c = ladder_idx(st, ps.HW)
    m = out[key]
    for P in [4.0, 10.0]:
        rr = [r for r in Hs if r["design"] == d and r["P"] == P]
        best = max(rr, key=lambda r: r["eta"]); hw = ps.HW + best["dhw"]
        d12 = (Et[b, a] - hw) / 0.010; d13 = (Et[c, a] - 2 * hw) / 0.010; d23 = d13 - d12
        pval = m["pcoh"] * min(1 + d12 ** 2, (1 + d23 ** 2) / m["pars"]["r23"])
        res0 = min(rr, key=lambda r: abs(r["dhw"]))
        # P_s at the detuned wavelength: same stack-averaged I_s (independent of hw), confinement factor of the
        # waveguide recomputed at that wavelength (hw_scan_wg.json); p_at_P = stack-averaged I/I_s at P
        wl = json.load(open(os.path.join(here, "hw_scan_wg.json")))[f"{best['dhw']:.4f}"]
        Ps_lam = m["Is_MWcm2"] * 1e10 * wl["t_stack_um"] * 1e-6 * wl["w_um"] * 1e-6 / wl["G1"]
        val[f"{d}|{P:g}"] = dict(dhw_meV=best["dhw"] * 1e3, eta=best["eta"], eta_res=res0["eta"], gain=best["eta"] / res0["eta"],
                                 d12=d12, d13=d13, d23=d23, P_valid_W=pval * m["Ps_W"], L_um=best["L_opt_um"],
                                 lam_um=wl["lam1_um"], G1_lam=wl["G1"], Ps_W_lam=Ps_lam, P_valid_W_lam=pval * Ps_lam,
                                 p_at_P=P / Ps_lam)
out["detuned_validity"] = val
json.dump(out, open(os.path.join(here, "numbers.json"), "w"), indent=1)
print(json.dumps(val, indent=1))

# Full three-level density matrix (dm_check.DM, resonant index compensated at the input intensity, and at the local
# intensity) against the rate-equation ladder at these detuned operating points: mapped ladder of each structure with
# the detunings, confinement factors and background losses of the waveguide at the detuned wavelength, p0 = p_at_P.
from model3 import Ladder
from dm_check import DM
WGS = json.load(open(os.path.join(here, "hw_scan_wg.json")))
dmop = {}
for d in ["ad", "tw"]:
    key = f"{d}_5e+11"; mp = V["maps"][key]; w0 = WGS["0.0000"]
    for P in [4.0, 10.0]:
        v = val[f"{d}|{P:g}"]; wl = WGS[f"{v['dhw_meV'] / 1e3:.4f}"]
        pars = dict(mp["pars"])
        pars.update(d12=v["d12"], d13=v["d13"], pcoh=out[key]["pcoh"],
                    u0=pars["u0"] * (wl["G1"] / wl["G2"]) / (w0["G1"] / w0["G2"]),
                    b1=wl["abg1_cm"] / (wl["G1"] * mp["A12_cm"]), b2=wl["abg2_cm"] / (wl["G1"] * mp["A12_cm"]))
        lad = Ladder(**{k: x for k, x in pars.items() if k != "pcoh"}, eta_s=pars["r23"] / (4 * pars["pcoh"]))
        r = lad.propagate(v["p_at_P"], npts=4000)
        xmax = min(3.0 * r["Lopt"] + 5.0, r["x"][-1])
        e_in = DM(**pars).propagate(v["p_at_P"], xmax=xmax); e_loc = DM(**pars).propagate(v["p_at_P"], xmax=xmax, comp="local")
        dmop[f"{d}|{P:g}"] = dict(pars=pars, p0=v["p_at_P"], eta_re=float(r["eta"]), ratio=e_in["eta"] / float(r["eta"]),
                                  ratio_local=e_loc["eta"] / float(r["eta"]), dm_at_end=e_in["end"])
# resonant 10 W operating point of Ref. OwnbyN1: dm_check ratio (ADQW-like, TW-like ladders) interpolated at 10 W / P_s
DMC = json.load(open(os.path.join(here, "dm_check.json")))
for d, name in [("ad", "ADQW-like"), ("tw", "TW-like")]:
    rr = sorted([r for r in DMC if r["tag"] == f"res|{name}"], key=lambda r: r["p0"])
    p10 = 10.0 / out[f"{d}_5e+11"]["Ps_W"]
    dmop[f"{d}|res10W"] = dict(p0=p10, ratio=float(np.interp(np.log(p10), np.log([r["p0"] for r in rr]), [r["ratio"] for r in rr])))
out["dm_oppoints"] = dmop
json.dump(out, open(os.path.join(here, "numbers.json"), "w"), indent=1)
print("dm_oppoints", json.dumps({k: {kk: x for kk, x in v.items() if kk != "pars"} for k, v in dmop.items()}, indent=1))

# Sec. 5: phase drift of chi(2) along the saturation curves (local S(rho(p)) / S(rho0), harmonic absent, p over the
# range of Fig. 3a), monotonic growth, and local maxima of the thin-film curve of Fig. 3b on a fine grid
from model3 import Ladder
from study_universal import CASES
satx = {}
pf = np.logspace(-2, 3, 4001)
for s in U["sat"]:
    lad = Ladder(**CASES[s["name"]], eta_s=1e-6)
    S = np.array([lad.coeffs(lad.rho(p))[2] / lad.S0 for p in pf])
    ph = np.degrees(np.unwrap(np.angle(S)))
    film = np.abs(S) ** 2 * pf
    loc = [i for i in range(1, len(pf) - 1) if film[i] >= film[i - 1] and film[i] >= film[i + 1]]
    t31, t32, r23 = lad.t31, lad.t32, lad.r23
    Cinf = (2 / 3) * ((1 + t31) / (lad.l12 * lad.D12) - (t32 + t31) / (r23 * lad.l23 * lad.D23)) / lad.D13
    satx[s["name"]] = dict(phase_drift_max_deg=float(np.max(np.abs(ph))),
                           phase_asym_deg=float(np.degrees(np.angle(Cinf / lad.S0))) if abs(Cinf) > 1e-12 else None,
                           sign_change=bool(np.max(np.abs(ph)) > 90), eta_monotonic=bool(np.all(np.diff(s["eta"]) > 0)),
                           film_local_peaks=[dict(p=float(pf[i]), value=float(film[i])) for i in loc])
out["sat_extra"] = satx
print("sat_extra", json.dumps(satx, indent=1))

# Sec. 8 (limitations): depolarisation shifts of the ladder transitions (Et - bare), the TW transition from the
# intermediate state (index 1) to index 4 against 2 hw, and the change of the transition energies when the Hartree
# potential and the depolarisation shifts are recomputed with the three ladder populations equalised (rho = 1/3,
# the large-intensity limit), using the qwkit building blocks of sp.self_consistent with fixed populations.
from qwkit import sp, core as qc
import tables as n1tables


def sc_states(layers, F, ns, ladder=None, spacer=200.0, dz=0.5, T=300.0, m_dos=0.043, mix=0.4, tol=1e-6, maxit=400):
    """As sp.self_consistent + sp.prepare + sp.depol_energies, but with the populations of the ladder states
    (prepared-state indices) fixed at 1/3 when `ladder` is given (Fermi-Dirac otherwise). Returns E, Et (eV)."""
    pad = spacer / 2.0
    z, V0, mz, Eg, active = qc.build(layers, pad=pad, dz=dz, pad_material="barrier")
    Vext = qc.potential(z, V0, active, F)
    nd = sp.doping_profile(z, active, layers, pad, ns, "spacer"); eps = sp.eps_profile(z, layers, pad, "barrier")
    VH = np.zeros_like(z); Eprev = None

    def pops(E, psi):
        if ladder is None:
            return qc.populations(E, ns, T, m_eff=m_dos)
        rho = np.zeros(len(E)); rho[np.flatnonzero(qc.localized(z, psi, active))[list(ladder)]] = 1.0 / 3.0
        return rho

    for _ in range(maxit):
        E, psi = qc.solve_states(z, Vext + VH, mz, Eg, nstates=7, E_guess=Eprev, tol=1e-6); Eprev = E
        VHn = sp.hartree(z, nd, (psi ** 2) @ (pops(E, psi) * ns), eps)
        err = np.max(np.abs(VHn - VH)); VH = (1 - mix) * VH + mix * VHn
        if err < tol:
            break
    E, psi = qc.solve_states(z, Vext + VH, mz, Eg, nstates=7, E_guess=Eprev)
    Ek, psik, _, rhoF = sp.prepare(dict(z=z, psi=psi, E=E, active=active, ns=ns, T=T, m_dos=m_dos))
    rho = rhoF if ladder is None else np.where(np.isin(np.arange(len(Ek)), list(ladder)), 1.0 / 3.0, 0.0)
    return Ek, sp.depol_energies(z, psik, Ek, rho, ns, eps_r=13.5)


shifts = {}
st4 = ps.T["states"][4]
for d, lay, F in [("ad", n1tables.AD, 0.0), ("tw", n1tables.TW, float(st4["F_tw"]))]:
    st = st4[d]; E = np.array(st["E"]); Et = np.array(st["Et"]); a, b, c = ladder_idx(st, ps.HW)
    tr = {"12": (b, a), "13": (c, a), "23": (c, b)}
    dep = {k: float((Et[i, j] - (E[i] - E[j])) * 1e3) for k, (i, j) in tr.items()}
    Eeq, Eteq = sc_states(lay, F, st4["ns"])
    Esat, Etsat = sc_states(lay, F, st4["ns"], ladder=(a, b, c))
    chg = {k: float((Etsat[i, j] - Eteq[i, j]) * 1e3) for k, (i, j) in tr.items()}
    shifts[d] = dict(ladder=[a, b, c], depol_meV=dep, depol_max_meV=max(abs(v) for v in dep.values()),
                     repro_Et_meV=float(np.max(np.abs(Eteq - Et)) * 1e3), sat_change_meV=chg,
                     sat_change_max_meV=max(abs(v) for v in chg.values()))
    if d == "tw":
        shifts[d]["E41_minus_2hw_meV"] = float((Et[4, 1] - 2 * ps.HW) * 1e3)
out["shifts"] = shifts
json.dump(out, open(os.path.join(here, "numbers.json"), "w"), indent=1)
print("shifts", json.dumps(shifts, indent=1))
