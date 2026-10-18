"""Large-intensity asymptote (Eq. asym) against numerical propagation at p0 = 3000 without pump depletion.
Writes asym_check.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
from model3 import Ladder
from study_universal import CASES, asymptote
out = {}
for name in ["ADQW-like", "TW-like", "weak 2-3"]:
    m = Ladder(**CASES[name], eta_s=1e-7)
    s = asymptote(m)[0]
    r = m.propagate(3000.0, npts=20000)
    out[name] = dict(eta_num=r["eta"] / m.eta_s, eta_asym=s * 3000.0, rel_dev=abs(r["eta"] / m.eta_s / (s * 3000.0) - 1))
    print(name, out[name])
json.dump(out, open(os.path.join(here, "asym_check.json"), "w"), indent=1)
