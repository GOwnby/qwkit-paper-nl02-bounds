"""Independent check of the rate-equation ladder model with the full steady-state three-level density matrix
(rotating-wave approximation, no perturbative expansion in the fields, all coherences including the two-photon
coherence rho_31, Autler-Townes splitting and power broadening included).

Rotating frame: H = diag(0, d12, d13) - [mu E] couplings, energies in units of gamma (HWHM, equal for all
coherences), time in hbar/gamma. With |e1|^2 = I1/Is and |e2|^2 = I2/Is:
   mu12 E1/gamma = e1 / (2 sqrt(pcoh)),  mu23 E1/gamma = sqrt(r23) e1 / (2 sqrt(pcoh)),
   mu13 E2/gamma = e2 / (2 sqrt(pcoh u0)),       pcoh = gamma tau21 / hbar.
Population relaxation (units gamma/hbar): 2->1 1/pcoh, 3->2 t32/pcoh, 3->1 t31/pcoh; coherences decay at 1.
Field equations (x in units of 1/A12):
   de1/dx = i sqrt(pcoh) (rho21 + sqrt(r23) rho32) - b1/2 e1
   de2/dx = i (2 sqrt(pcoh)/sqrt(u0)) rho31       - b2/2 e2
In the weak-field limit these reproduce A12, A13 and chi(2) of the ladder model exactly. Writes dm_check.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import numpy as np
from multiprocessing import Pool
from scipy.integrate import solve_ivp
from model3 import Ladder
from study_universal import CASES
from detuning_study import d13_rule

IDX = [(i, j) for i in range(3) for j in range(3)]


class DM:
    def __init__(self, r23=1.0, u0=4.0, d12=0.0, d13=0.0, t32=1.0, t31=1.0, b1=0.0, b2=0.0, pcoh=15.0):
        self.r23, self.u0, self.d12, self.d13, self.t32, self.t31 = r23, u0, d12, d13, t32, t31
        self.b1, self.b2, self.pcoh = b1, b2, pcoh
        self.H0 = np.diag([0.0, d12, d13]).astype(complex)

    def steady(self, e1, e2):
        s = 2 * np.sqrt(self.pcoh)
        V = np.zeros((3, 3), complex)
        V[1, 0] = -e1 / s; V[2, 1] = -np.sqrt(self.r23) * e1 / s; V[2, 0] = -e2 / (s * np.sqrt(self.u0))
        H = self.H0 + V + V.conj().T - np.diag(np.diag(V + V.conj().T))
        k21, k32, k31 = 1 / self.pcoh, self.t32 / self.pcoh, self.t31 / self.pcoh
        L = np.zeros((9, 9), complex)
        for n, (a, b) in enumerate(IDX):
            E = np.zeros((3, 3), complex); E[a, b] = 1.0
            D = -1j * (H @ E - E @ H)
            # relaxation
            if a == b:
                if a == 1:
                    D[1, 1] -= k21; D[0, 0] += k21
                if a == 2:
                    D[2, 2] -= k32 + k31; D[1, 1] += k32; D[0, 0] += k31
            else:
                D[a, b] -= 1.0
            L[:, n] = D.reshape(9)
        rhs = np.zeros(9, complex)
        L[0, :] = 0; L[0, 0] = L[0, 4] = L[0, 8] = 1.0; rhs[0] = 1.0
        return np.linalg.solve(L, rhs).reshape(3, 3)

    def phase_rates(self, p):
        """Linear phase rates (rad per 1/A12) of the pump and of a weak harmonic at pump intensity p: the
        resonant refractive index, which a phase-matched design compensates at its design intensity."""
        c1 = np.sqrt(self.pcoh); c2 = 2 * np.sqrt(self.pcoh) / np.sqrt(self.u0)
        e1 = np.sqrt(max(p, 1e-12)) + 0j
        r = self.steady(e1, 0.0)
        k1 = 1j * c1 * (r[1, 0] + np.sqrt(self.r23) * r[2, 1]) / e1
        de = 1e-6 * (1 + abs(e1))
        r2 = self.steady(e1, de)
        k2 = 1j * c2 * (r2[2, 0] - r[2, 0]) / de
        return float(k1.imag), float(k2.imag)

    def propagate(self, p0, xmax, npts=3000, rtol=1e-7, comp="input"):
        """comp: 'input' compensates the resonant index at the input intensity (design point = operating
        power), 'ss' at small signal, None leaves it uncompensated."""
        c1 = np.sqrt(self.pcoh); c2 = 2 * np.sqrt(self.pcoh) / np.sqrt(self.u0)
        ph1, ph2 = (0.0, 0.0) if comp in (None, "local") else self.phase_rates(p0 if comp == "input" else 1e-8)

        def rhs(x, y):
            e1 = y[0] + 1j * y[1]; e2 = y[2] + 1j * y[3]
            r = self.steady(e1, e2)
            q1, q2 = self.phase_rates(abs(e1) ** 2) if comp == "local" else (ph1, ph2)   # chirped: local intensity
            d1 = 1j * c1 * (r[1, 0] + np.sqrt(self.r23) * r[2, 1]) - self.b1 / 2 * e1 - 1j * q1 * e1
            d2 = 1j * c2 * r[2, 0] - self.b2 / 2 * e2 - 1j * q2 * e2
            return [d1.real, d1.imag, d2.real, d2.imag]
        xs = np.linspace(0, xmax, npts)
        sol = solve_ivp(rhs, (0, xmax), [np.sqrt(p0), 0, 0, 0], t_eval=xs, rtol=rtol, atol=1e-13 * (1 + p0), method="DOP853")
        q = sol.y[2] ** 2 + sol.y[3] ** 2; i = int(np.argmax(q))
        return dict(eta=float(q[i] / p0), Lopt=float(sol.t[i]), end=bool(i > 0.98 * len(q)))


def job(args):
    tag, pars, p0 = args
    lad = Ladder(**{k: v for k, v in pars.items() if k != "pcoh"}, eta_s=pars["r23"] / (4 * pars["pcoh"]))
    r = lad.propagate(p0, npts=4000)
    xmax = min(3.0 * r["Lopt"] + 5.0, r["x"][-1])
    dm = DM(**pars).propagate(p0, xmax=xmax)
    if dm["end"]:
        dm = DM(**pars).propagate(p0, xmax=3 * xmax)
    dss = DM(**pars).propagate(p0, xmax=xmax, comp="ss")
    d12, d23 = pars.get("d12", 0.0), pars.get("d13", 0.0) - pars.get("d12", 0.0)
    pval = pars["pcoh"] * min(1 + d12 ** 2, (1 + d23 ** 2) / pars["r23"])
    return dict(tag=tag, pars=pars, p0=p0, eta_re=float(r["eta"]), eta_dm=dm["eta"], ratio=dm["eta"] / r["eta"], eta_dm_ss=dss["eta"],
                L_re=float(r["Lopt"]), L_dm=dm["Lopt"], p_valid=float(pval))


if __name__ == "__main__":
    # weak-field consistency
    dm = DM(r23=1.7, u0=3.7, d12=0.3, d13=-0.5, pcoh=15.0)
    lad = Ladder(r23=1.7, u0=3.7, d12=0.3, d13=-0.5, eta_s=1.7 / 60)
    print("weak-field check (eta_dm/eta_re):", dm.propagate(1e-3, 30)["eta"] / lad.propagate(1e-3, npts=4000)["eta"], flush=True)
    jobs = []
    ps = list(np.logspace(-1, 2.5, 15))
    for name in ["balanced", "ADQW-like", "TW-like", "weak 2-3"]:
        pars = dict(CASES[name]); pars["pcoh"] = 15.0
        jobs += [(f"res|{name}", pars, float(p)) for p in ps]
    for p in [10.0, 100.0]:                         # detuning rule (ADQW-like)
        for d12 in [-8.0, -6.0, -4.0, -2.0]:
            pars = dict(CASES["ADQW-like"]); pars.update(d12=d12, d13=d13_rule(pars["u0"], d12), b1=1e-4, b2=1e-4, pcoh=15.0)
            jobs.append((f"det|{p:g}", pars, p))
    with Pool(2) as pool:
        res = pool.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(here, "dm_check.json"), "w"), indent=1)
    for r in res:
        print(f"{r['tag']:16s} p0={r['p0']:8.2f} (valid<{r['p_valid']:.1f}) d12={r['pars'].get('d12',0):+.1f} "
              f"eta_re={r['eta_re']:.4e} eta_dm={r['eta_dm']:.4e} ratio={r['ratio']:.3f}")
