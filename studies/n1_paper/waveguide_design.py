"""Waveguide for the N1 device: TM00 at 10 um and 5 um, overlap with the QW stack, modal coherence length,
background loss, and the bias voltage the stack needs. Writes waveguide.json."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np
from qwkit import waveguide as wg
NPER, LPER_A = 40, 378.0                       # periods, period length (58+14+34+14+58+200 A)
T_STACK = NPER * LPER_A * 1e-4                 # um
FRAC_WELL = 150.0 / 378.0
def stack_layers(t_up, t_sp, ns=5e11, sub_N=1e17):
    Nst = ns / (LPER_A * 1e-8)
    return [dict(material="Au", t_um=0.1),
            dict(material="InP", t_um=t_up),                      # Fe-doped semi-insulating
            dict(material="InGaAs", t_um=t_sp),
            dict(material="stack", t_um=T_STACK, N=0.0, frac_well=FRAC_WELL),   # z-polarised response of confined electrons = ISB, handled in propagation
            dict(material="InGaAs", t_um=t_sp),
            dict(material="InP", t_um=12.0, N=sub_N)]
def analyse(t_up, t_sp, ns=5e11, dz=0.01):
    L = stack_layers(t_up, t_sp, ns)
    m1 = wg.tm_modes(L, 10.0, dz, 1)[0]; m2 = wg.tm_modes(L, 5.0, dz, 1)[0]
    z0 = 0.1 + t_up + t_sp; z1 = z0 + T_STACK
    G1 = wg.overlap(m1, z0, z1); G2 = wg.overlap(m2, z0, z1)
    O = wg.shg_overlap(m1, m2, z0, z1)                         # um^-1
    dn = m2["neff"].real - m1["neff"].real
    Lcoh = 10.0 / (4 * dn)
    V = 55e3 * 1e-4 * (t_up + 2 * t_sp + T_STACK)             # V at 55 kV/cm (uniform-field estimate)
    return dict(t_up=t_up, t_sp=t_sp, n1=m1["neff"].real, n2=m2["neff"].real, a1=m1["alpha_cm"], a2=m2["alpha_cm"],
                G1=G1, G2=G2, O_um=O, Lcoh_um=Lcoh, V55=V, z0=z0, z1=z1,
                prof=dict(z=m1["z"].tolist(), Ez1=np.abs(m1["Ez"]).tolist(), Ez2=np.abs(m2["Ez"]).tolist(),
                          eps1=m1["eps"].real.tolist()))
if __name__ == "__main__":
    res = []
    for t_up, t_sp in ((2.0, 0.6), (2.0, 0.9), (3.0, 0.9)):
            r = analyse(t_up, t_sp); res.append(r)
            print(f"t_up {t_up} t_sp {t_sp}: n1 {r['n1']:.4f} n2 {r['n2']:.4f} Lcoh {r['Lcoh_um']:.1f} um | "
                  f"G1 {r['G1']:.3f} G2 {r['G2']:.3f} O {r['O_um']:.4f}/um | loss {r['a1']:.2f}/{r['a2']:.2f} cm^-1 | V {r['V55']:.1f}", flush=True)
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "waveguide.json"), "w"))
