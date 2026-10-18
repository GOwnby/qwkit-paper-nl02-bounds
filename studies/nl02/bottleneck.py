"""Triple well with a 10x faster decay of its intermediate (bottleneck) state, 1 -> 0 and 0 -> 1 rates scaled
(detailed balance kept). Writes bottleneck.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "..")); sys.path.insert(0, os.path.join(here, "..", "n1_paper")); sys.path.insert(0, here)
import numpy as np
from multiprocessing import Pool
from qwkit import propagation as pr
import propagation_study as ps
from qwkit_validation import R_for, GAM

def job(args):
    i, f, P = args
    st = ps.T["states"][i]["tw"]; R = R_for(i, "tw").copy(); R[1, 0] *= f; R[0, 1] *= f
    med = pr.Medium(st, ps.HW, gamma=GAM, R=R); w = ps.wg_for(st)
    r = pr.propagate(med, w, P, "perfect", L_max_um=3000.0 if P < 30 else 8000.0, steps_per_lcoh=16)
    return dict(i=i, f=f, P=float(P), eta=float(r["eta"]), L_opt_um=float(r["L_opt_um"]),
                tau1_ps=float(1e12 / R[1].sum()))

if __name__ == "__main__":
    Ps = list(np.logspace(-2, np.log10(300), 22))
    with Pool(2) as pool:
        res = pool.map(job, [(4, f, P) for f in [3.0, 10.0] for P in Ps], chunksize=2)
    json.dump(res, open(os.path.join(here, "bottleneck.json"), "w"))
    for r in res:
        if abs(np.log10(r["P"]) - 1) < 0.05 or abs(np.log10(r["P"]) - 0) < 0.12:
            print(r)
