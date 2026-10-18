"""Density matrix with the resonant index compensated at the local intensity (chirped phase matching) for the
detuned-rule points and two resonant ladders at high intensity. Writes dm_local.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
from multiprocessing import Pool
from dm_check import DM
D = json.load(open(os.path.join(here, "dm_check.json")))
def job(r):
    xmax = 3.0 * r["L_re"] + 5.0
    d = DM(**r["pars"]).propagate(r["p0"], xmax=xmax, comp="local")
    return dict(tag=r["tag"], d12=r["pars"].get("d12", 0.0), p0=r["p0"], ratio_input=r["ratio"], ratio_local=d["eta"] / r["eta_re"])
if __name__ == "__main__":
    sel = [r for r in D if r["tag"].startswith("det")] + [r for r in D if r["tag"] in ("res|ADQW-like", "res|TW-like") and r["p0"] > 300]
    with Pool(2) as pool:
        out = pool.map(job, sel)
    json.dump(out, open(os.path.join(here, "dm_local.json"), "w"), indent=1)
    for o in out: print(o)
