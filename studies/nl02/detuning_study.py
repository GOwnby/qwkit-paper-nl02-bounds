"""Escaping saturation by detuning: eta/(p eta_s) vs pump detuning d12 at fixed pump intensity p, with the
SH detuning chosen by the rule u = 1 (d13 = -sign(d12) sqrt(1/(u0 l12) - 1) when u0 l12 < 1, else 0).
Compared with Theorem 1 (e^-2 zeta1 zeta2 at u = 1).  Writes detuning.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import numpy as np
from multiprocessing import Pool
from model3 import Ladder
from study_universal import CASES


def d13_rule(u0, d12):
    l12 = 1 / (1 + d12 ** 2)
    if u0 * l12 >= 1:
        return 0.0
    return float(-np.sign(d12 if d12 != 0 else 1) * np.sqrt(1 / (u0 * l12) - 1))


def job(args):
    name, p, d12, b = args
    pars = dict(CASES[name]); pars.update(d12=d12, d13=d13_rule(pars["u0"], d12), b1=b, b2=b)
    m = Ladder(**pars, eta_s=1e-6)
    r = m.propagate(p, npts=3000)
    return dict(name=name, p=p, d12=d12, d13=pars["d13"], b=b, x=float(r["eta"] / (p * m.eta_s)),
                ss=float(m.eta_small_signal(p) / (p * m.eta_s)), L=float(r["Lopt"]))


if __name__ == "__main__":
    d12s = list(np.linspace(-8, 8, 33))
    jobs = [(n, p, float(d), b) for n in ["ADQW-like", "TW-like"] for p in [1.0, 10.0, 100.0] for d in d12s for b in [1e-4, 1e-2]]
    with Pool(2) as pool:
        res = pool.map(job, jobs, chunksize=4)
    json.dump(res, open(os.path.join(here, "detuning.json"), "w"))
    for n in ["ADQW-like", "TW-like"]:
        for p in [1.0, 10.0, 100.0]:
            for b in [1e-4, 1e-2]:
                rr = [r for r in res if r["name"] == n and r["p"] == p and r["b"] == b]
                best = max(rr, key=lambda r: r["x"]); res0 = min(rr, key=lambda r: abs(r["d12"]))
                print(f"{n:10s} p={p:5.0f} b={b:.0e}: at d12=0 x={res0['x']:.4f}; best x={best['x']:.4f} at d12={best['d12']:.1f} "
                      f"d13={best['d13']:.2f} L={best['L']:.0f} (Thm1 there {best['ss']:.4f})")
