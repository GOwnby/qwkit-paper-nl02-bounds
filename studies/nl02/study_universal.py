"""Universal-ladder studies for the NL-02 paper. Writes universal.json.
 1. Theorem 1 check (unsaturated propagation vs closed form) over random ladders.
 2. Theorem 2 check: random ladders, saturation on, random pump power, background loss, phase mismatch / QPM.
 3. Saturation curves eta/eta_s vs p for representative ladders, with the large-p asymptote.
 4. Thin-film (metasurface) counterpart |S(p)|^2 p.
 5. Achievable efficiency inside the rate-equation regime vs detunings (design maps)."""
import os, sys, json, time
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import numpy as np
from multiprocessing import Pool
from model3 import Ladder, g_of_u, L_opt
from scipy.optimize import minimize_scalar

def _mapped(key):
    """Ladder parameters mapped from the full qwkit structure at n_s = 1e11 cm^-2 (qwkit_validation.json, written by
    qwkit_validation.py; run that first), rounded to two decimals."""
    m = json.load(open(os.path.join(here, "qwkit_validation.json")))["maps"][key]["pars"]
    return {k: round(float(m[k]), 2) for k in ("r23", "u0", "d12", "d13", "t32", "t31")}


CASES = {
    "balanced": dict(r23=1.0, u0=4.0, t32=1.0, t31=1.0),
    "ADQW-like": _mapped("ad_1e+11"),
    "TW-like": _mapped("tw_1e+11"),
    "weak 2-3": dict(r23=0.3, u0=4.0, t32=1.0, t31=1.0),
}


def asymptote(m):
    """Large-p slope: eta/eta_s -> s_inf * p (no background loss); returns s_inf, C_inf, k_inf, nu."""
    t31, t32, r23 = m.t31, m.t32, m.r23
    C = (2 / 3) * ((1 + t31) / (m.l12 * m.D12) - (t32 + t31) / (r23 * m.l23 * m.D23)) / m.D13
    k = (2 / 3) * ((1 + t31) + (t32 + t31))
    mm = (2 / m.u0) * m.l13 * (2 / 3) * ((1 + t31) / m.l12 + (t32 + t31) / (r23 * m.l23))
    nu = mm / (2 * k)
    f = lambda y: -(y ** nu * (1 - y ** (1 - nu)) / (1 - nu))
    h = -minimize_scalar(f, bounds=(1e-12, 1), method="bounded").fun
    s = abs(C) ** 2 * h ** 2 / (m.u0 * k ** 2) * (m.c ** 2 * m.u0 / m.eta_s)
    return float(s), float(abs(C)), float(k), float(nu)


def rand_ladder(rng, sat=True):
    return dict(r23=float(10 ** rng.uniform(-1, 1)), u0=float(10 ** rng.uniform(-0.7, 1.5)),
                d12=float(rng.uniform(-3, 3)), d13=float(rng.uniform(-3, 3)),
                t32=float(10 ** rng.uniform(-1, 1)), t31=float(10 ** rng.uniform(-1, 1)),
                b1=float(rng.choice([0, 10 ** rng.uniform(-3, 0)])), b2=float(rng.choice([0, 10 ** rng.uniform(-3, 0)])),
                eta_s=float(10 ** rng.uniform(-4, -1.3)))


def job_t1(seed):
    rng = np.random.default_rng(seed)
    pars = rand_ladder(rng); pars["eta_s"] = 1e-7
    m = Ladder(**pars)
    r = m.propagate(1e-3, saturate=False, npts=8000)
    return float(r["eta"] / m.eta_small_signal(1e-3))


def job_t2(seed):
    rng = np.random.default_rng(1000 + seed)
    pars = rand_ladder(rng)
    m = Ladder(**pars)
    mode = rng.choice(["pm"] * 7 + ["dk", "sign", "onoff"])
    p0 = float(10 ** rng.uniform(-2, 2.5 if mode == "pm" else 1.5))
    dk = 0.0 if mode == "pm" else float(10 ** rng.uniform(-0.5, 0.7))
    r = m.propagate(p0, npts=3000, dk=dk, qpm=None if mode in ("pm", "dk") else mode)
    # population-weighted bound: sup over the populations met along the device (check no inversion)
    rhos = [m.rho(p, 0.0) for p in np.linspace(min(r["p"]), p0, 60)]
    inv = any(rr[1] > rr[0] + 1e-9 or rr[2] > rr[1] + 1e-9 for rr in rhos)
    return dict(pars=pars, p0=p0, mode=str(mode), dk=dk, eta=float(r["eta"]), bound=float(m.bound2(p0)),
                ratio=float(r["eta"] / m.bound2(p0)), inversion=bool(inv))


def sat_curve(name, pgrid):
    m = Ladder(**CASES[name], eta_s=1e-6)
    et = [m.propagate(p, npts=6000)["eta"] / m.eta_s for p in pgrid]
    film = []
    for p in pgrid:                       # thin film, fixed thickness: eta_ms / (eta_s A1 A2 / 2) = |S/S0|^2 p
        rho = m.rho(p)
        _, _, S = m.coeffs(rho)
        film.append(float(abs(S / m.S0) ** 2 * p))
    s_inf = asymptote(m)
    return dict(name=name, pars=CASES[name], p=list(map(float, pgrid)), eta=list(map(float, et)),
                ss=[float(m.eta_small_signal(p) / m.eta_s) for p in pgrid], bound=[float(m.bound2(p) / m.eta_s) for p in pgrid],
                film=film, s_inf=s_inf[0], C_inf=s_inf[1], k_inf=s_inf[2], nu=s_inf[3])


def map_job(args):
    name, d12, d13, pcoh, b = args
    base = dict(CASES[name]); base.update(d12=d12, d13=d13, b1=b, b2=b)
    r23 = base["r23"]
    eta_s = r23 / (4 * pcoh)
    m = Ladder(**base, eta_s=eta_s)
    pmax = pcoh * min(1 + d12 ** 2, (1 + (d13 - d12) ** 2) / r23)
    best = (0.0, 0.0, 0.0)
    for p in np.logspace(-1, np.log10(pmax), 14):
        r = m.propagate(p, npts=2500)
        if r["eta"] > best[0]:
            best = (float(r["eta"]), float(p), float(r["Lopt"]))
    return dict(name=name, d12=d12, d13=d13, eta=best[0], p=best[1], L=best[2], pmax=float(pmax))


if __name__ == "__main__":
    t0 = time.time(); out = {}
    with Pool(2) as pool:
        t1 = pool.map(job_t1, range(300))
        out["theorem1"] = dict(n=len(t1), min=min(t1), max=max(t1))
        print("Theorem 1 ratios", out["theorem1"], time.time() - t0, flush=True)
        t2 = pool.map(job_t2, range(2000), chunksize=10)
        out["theorem2"] = t2
        rat = [x["ratio"] for x in t2]
        print("Theorem 2: max ratio", max(rat), "median", np.median(rat), "inversions", sum(x["inversion"] for x in t2),
              time.time() - t0, flush=True)
        pgrid = np.logspace(-2, 3, 41)
        out["sat"] = pool.starmap(sat_curve, [(n, pgrid) for n in CASES])
        for s in out["sat"]:
            i = int(np.argmax(np.array(s["eta"]) * (np.array(s["p"]) <= 20)))
            print(s["name"], "peak(p<=20)", s["eta"][i], "at p", s["p"][i], "s_inf", s["s_inf"], "nu", s["nu"],
                  "film peak", max(s["film"]), flush=True)
        dgrid = np.linspace(-3, 3, 13)
        jobs = [(n, float(a), float(b), 15.0, 1e-3) for n in ["ADQW-like", "TW-like"] for a in dgrid for b in dgrid]
        out["maps"] = pool.map(map_job, jobs, chunksize=4)
        for n in ["ADQW-like", "TW-like"]:
            mm = [x for x in out["maps"] if x["name"] == n]
            b = max(mm, key=lambda x: x["eta"])
            r0 = min(mm, key=lambda x: abs(x["d12"] - CASES[n].get("d12", 0)) + abs(x["d13"] - CASES[n].get("d13", 0)))
            print(n, "best", b, "near nominal", r0, flush=True)
    out["g_of_u"] = dict(u=list(np.logspace(-2, 2, 81)), g=list(map(float, g_of_u(np.logspace(-2, 2, 81)))))
    json.dump(out, open(os.path.join(here, "universal.json"), "w"))
    print("done", time.time() - t0)
