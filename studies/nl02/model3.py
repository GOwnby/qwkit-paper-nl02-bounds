"""
model3.py -- dimensionless three-level ladder model of resonant intersubband SHG with saturation.

Levels 1 < 2 < 3, pump at w drives 1-2 (detuning d12 = E21 - hw) and 2-3 (d23 = E32 - hw = d13 - d12),
SH at 2w drives 1-3 (d13 = E31 - 2hw).  All detunings in units of the (common) linewidth gamma (HWHM).
Boyd field convention, rate-equation (incoherent) regime, plane waves, perfect phase matching.

Units:
  x~ = A12 x            A12 = resonant 1-2 absorption with all electrons in level 1
  p  = I1 / Is          Is  = hbar w / (2 sigma12_res tau21)  (resonant two-level saturation intensity)
  q  = I2 / Is
  rates in 1/tau21
Parameters:
  r23  = (z23/z12)^2            u0 = 2 A12 / A13 = (z12/z13)^2 (n2/n1)
  t32  = tau21/tau32,  t31 = tau21/tau31
  b1, b2 = background modal losses in units of A12
  eta_s = (hbar Omega23(Is))^2 / (4 gamma^2) = r23 * hbar / (4 tau21 gamma)   (efficiency scale)
Equations:
  de1/dx = -(a1(rho)+b1)/2 e1 + i conj(c S(rho)) e2 conj(e1)
  de2/dx = -(a2(rho)+b2)/2 e2 + i c S(rho) e1^2,       |c|^2 = eta_s / u0
  a1 = l12 (rho1-rho2) + r23 l23 (rho2-rho3),  a2 = (2/u0) l13 (rho1-rho3),  l = 1/(1+d^2)
  S  = [(rho1-rho2)/D12 - (rho2-rho3)/D23] / D13,   D = d - i
Small-signal, uniform theorem (Theorem 1):  eta(L*) = g(u) p eta_s zeta1 zeta2,  u = 2 a1t/a2t,
  g(u) = u^(-(u+1)/(u-1)) <= exp(-2).
General bound (Theorem 2): eta <= (p0 eta_s / 4) max_rho[(r23 (rho1-rho2) + (rho2-rho3)) / (r23 (rho1-rho3))]
  <= (p0 eta_s / 4) max(1, 1/r23)     (any saturation, depletion, length, detuning, no inversion).
"""
import numpy as np
from scipy.integrate import solve_ivp


def lor(d):
    return 1.0 / (1.0 + d * d)


def g_of_u(u):
    """Length-optimised small-signal factor: max_L [(e^{-aL}-e^{-bL})/(b-a)]^2 * a*b, u = a/b, a = alpha1,
    b = alpha2/2.  g(1) = e^-2."""
    u = np.asarray(u, dtype=float)
    out = np.where(np.abs(u - 1) < 1e-9, np.exp(-2.0), 0.0)
    m = np.abs(u - 1) >= 1e-9
    uu = u[m]
    out[m] = uu ** (-(uu + 1) / (uu - 1))
    return out if out.ndim else float(out)


def L_opt(a1, a2):
    """Optimum length for constant absorptions (pump power absorption a1, SH power absorption a2)."""
    a, b = a1, a2 / 2
    if abs(a - b) < 1e-12 * max(a, b):
        return 1.0 / a
    return np.log(a / b) / (a - b)


class Ladder:
    def __init__(self, r23=1.0, u0=4.0, d12=0.0, d13=0.0, t32=1.0, t31=1.0, b1=0.0, b2=0.0, eta_s=0.02, c=None):
        """c: optional explicit coupling (overrides sqrt(eta_s/u0)), e.g. a waveguide with overlap factor."""
        self.r23, self.u0, self.d12, self.d13 = r23, u0, d12, d13
        self.d23 = d13 - d12
        self.t32, self.t31, self.b1, self.b2, self.eta_s = t32, t31, b1, b2, eta_s
        self.l12, self.l13, self.l23 = lor(d12), lor(d13), lor(self.d23)
        self.D12, self.D13, self.D23 = d12 - 1j, d13 - 1j, self.d23 - 1j
        self.c = np.sqrt(eta_s / u0) if c is None else c
        self.S0 = 1.0 / (self.D12 * self.D13)          # S at rho = (1,0,0)

    def rho(self, p, q=0.0):
        W12 = self.l12 * p / 2; W23 = self.r23 * self.l23 * p / 2; W13 = self.l13 * q / (2 * self.u0)
        # rows: d rho1, d rho2 (rho3 = 1 - rho1 - rho2)
        # d1 = -W12(r1-r2) - W13(r1-r3) + r2 + t31 r3
        # d2 =  W12(r1-r2) - W23(r2-r3) - r2 + t32 r3
        M = np.zeros((3, 3)); bvec = np.zeros(3)
        M[0] = [-W12 - W13, W12 + 1.0, W13 + self.t31]
        M[1] = [W12, -W12 - W23 - 1.0, W23 + self.t32]
        M[2] = [1.0, 1.0, 1.0]; bvec[2] = 1.0
        return np.linalg.solve(M, bvec)

    def coeffs(self, rho):
        r1, r2, r3 = rho
        a1 = self.l12 * (r1 - r2) + self.r23 * self.l23 * (r2 - r3)
        a2 = (2.0 / self.u0) * self.l13 * (r1 - r3)
        S = ((r1 - r2) / self.D12 - (r2 - r3) / self.D23) / self.D13
        return a1, a2, S

    def small_signal(self):
        a1 = self.l12; a2 = (2.0 / self.u0) * self.l13
        return a1 + self.b1, a2 + self.b2

    def eta_small_signal(self, p):
        """Theorem 1 (exact for constant coefficients, undepleted)."""
        a1t, a2t = self.small_signal()
        u = 2 * a1t / a2t
        z1 = self.l12 / a1t; z2 = (2.0 / self.u0) * self.l13 / a2t
        return g_of_u(u) * p * self.eta_s * z1 * z2

    def propagate(self, p0, xmax=None, saturate=True, rtol=1e-8, npts=4000, dk=0.0, qpm=None):
        """Integrate; returns x, p(x), q(x) and eta = max q / p0.  dk: phase mismatch (units of A12);
        qpm: None, 'sign' or 'onoff' (coherence length pi/dk)."""
        if xmax is None:
            a1t, a2t = self.small_signal()
            Ls = L_opt(a1t, a2t)
            xmax = 12 * Ls
            if saturate:                     # a bleached pump penetrates ~ p0 / (saturated loss rate)
                xmax = max(xmax, 3 * (p0 + 1) / a1t + 6 * Ls)
        c = self.c

        def rhs(x, y):
            e1 = y[0] + 1j * y[1]; e2 = y[2] + 1j * y[3]
            p = abs(e1) ** 2; q = abs(e2) ** 2
            rho = self.rho(p, q) if saturate else np.array([1.0, 0.0, 0.0])
            a1, a2, S = self.coeffs(rho)
            cS = c * S                       # |c S0|^2 = eta_s l12 l13 / u0  (Theorem 1 normalisation)
            if dk != 0.0:
                sgn = 1.0
                if qpm is not None:
                    seg = int(np.floor(x * dk / np.pi)) % 2
                    sgn = (1.0 if seg == 0 else -1.0) if qpm == "sign" else (1.0 if seg == 0 else 0.0)
                cS = cS * sgn * np.exp(1j * dk * x)
            d1 = -(a1 + self.b1) / 2 * e1 + 1j * np.conj(cS) * e2 * np.conj(e1)
            d2 = -(a2 + self.b2) / 2 * e2 + 1j * cS * e1 ** 2
            return [d1.real, d1.imag, d2.real, d2.imag]

        xs = np.linspace(0, xmax, npts)
        mstep = np.inf if dk == 0.0 else np.pi / dk / 8
        sol = solve_ivp(rhs, (0, xmax), [np.sqrt(p0), 0, 0, 0], t_eval=xs, rtol=rtol, atol=1e-14 * (1 + p0),
                        method="DOP853", max_step=mstep)
        e1 = sol.y[0] + 1j * sol.y[1]; e2 = sol.y[2] + 1j * sol.y[3]
        p = abs(e1) ** 2; q = abs(e2) ** 2
        i = int(np.argmax(q))
        return dict(x=sol.t, p=p, q=q, eta=q[i] / p0, Lopt=sol.t[i], ok=sol.success, end_frac=q[-1] / max(q[i], 1e-300))

    def bound2(self, p0):
        """Theorem 2 bound (population-independent form)."""
        return p0 * self.eta_s / 4 * max(1.0, 1.0 / self.r23)
