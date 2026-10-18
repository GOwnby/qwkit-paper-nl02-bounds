"""Figures for the NL-02 manuscript, drawn only from the study JSON outputs. Writes ../../paper_nl02/figures/*.pdf."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model3 import g_of_u

outdir = os.path.join(here, "..", "..", "paper_nl02", "figures"); os.makedirs(outdir, exist_ok=True)
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a4fbf"
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#9b9a95"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.0, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "lines.linewidth": 1.5,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
V = json.load(open(os.path.join(here, "qwkit_validation.json")))
U = json.load(open(os.path.join(here, "universal.json")))
D = json.load(open(os.path.join(here, "detuning.json")))
B = json.load(open(os.path.join(here, "bottleneck.json")))
H = json.load(open(os.path.join(here, "hw_scan.json")))
N = json.load(open(os.path.join(here, "numbers.json")))
DM = json.load(open(os.path.join(here, "dm_check.json")))
COL = {"balanced": INK2, "ADQW-like": BLUE, "TW-like": ORANGE, "weak 2-3": AQUA}


def panel_label(ax, s, x=-0.17, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", color=INK)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)


def fig1():
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw=dict(wspace=0.42))
    ax = axs[0]
    u = np.logspace(-2, 2, 400)
    ax.plot(u, g_of_u(u), color=INK, label="with SH absorption, $g(u)$")
    ax.plot(u, 1 / u, color=MUTED, ls="--", lw=1.1, label="SH absorption neglected, $1/u$")
    ax.axhline(np.exp(-2), color=INK2, lw=0.6, ls=":")
    ax.text(0.012, np.exp(-2) * 1.12, "$e^{-2}$", color=INK2, fontsize=7.5)
    for key, c, lab in [("ad_5e+11", BLUE, "ADQW"), ("tw_5e+11", ORANGE, "TW")]:
        u0 = N[key]["pars"]["u0"]
        ax.plot([u0], [g_of_u(u0)], "o", color=c, ms=5, zorder=5)
        ax.annotate(lab, (u0, g_of_u(u0)), xytext=(-5, -12) if lab == "ADQW" else (-6, 6), textcoords="offset points",
                    color=c, fontsize=7.5, ha="right")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(5e-3, 3)
    ax.set_xlabel(r"$u=2\alpha_\omega/\alpha_{2\omega}$"); ax.set_ylabel(r"$\eta\,/\,[(\hbar\Omega_{23})^2/4\gamma_{12}\gamma_{13}]$")
    ax.legend(frameon=False, loc="upper right", fontsize=6.4); clean(ax); panel_label(ax, "a")
    for ax, key, title, lab in [(axs[1], "ad_5e+11", "ADQW", "b"), (axs[2], "tw_5e+11", "Triple well", "c")]:
        sc = V["identity"][key]["scan"]
        x = np.array([s["hw"] for s in sc]) * 1e3 - 123.984
        chi2 = np.array([s["chi"] for s in sc]) ** 2; aa = np.array([s["a1"] * s["a2"] for s in sc])
        rat = np.array([s["ratio"] for s in sc])
        ax.plot(x, chi2 / chi2.max(), color=BLUE, lw=1.2, label=r"$|\chi^{(2)}|^2$")
        ax.plot(x, aa / aa.max(), color=AQUA, lw=1.2, ls="--", label=r"$\alpha_\omega\alpha_{2\omega}$")
        ax.plot(x, rat, color=INK, lw=1.6, label=r"$|\chi^{(2)}|^2/(Q_{23}\alpha_\omega\alpha_{2\omega})$")
        ax.set_yscale("log"); ax.set_ylim(1e-4, 4); ax.set_xlim(-40, 40)
        ax.set_xlabel(r"$\hbar\omega-124$ meV"); ax.set_title(title, color=INK)
        if lab == "b":
            ax.set_ylabel("normalised")
            ax.legend(frameon=False, loc="lower left", fontsize=6.3)
        clean(ax); panel_label(ax, lab)
    fig.savefig(os.path.join(outdir, "fig1_identity.pdf")); plt.close(fig)


def fig2():
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    t2 = U["theorem2"]
    for mode, c, lab in [("pm", BLUE, "phase matched"), ("dk", AQUA, r"mismatched ($\Delta k\neq0$)"),
                         ("sign", ORANGE, "sign-reversal QPM"), ("onoff", PURPLE, "on/off QPM")]:
        xs = [x["p0"] for x in t2 if x["mode"] == mode]; ys = [x["ratio"] for x in t2 if x["mode"] == mode]
        ax.scatter(xs, ys, s=3, color=c, alpha=0.6, lw=0, label=lab)
    ax.axhline(1, color=INK, lw=1.0); ax.text(0.012, 1.15, "Theorem 2 bound", fontsize=7, color=INK)
    ax.axhline(4 * np.exp(-2), color=INK2, lw=0.7, ls="--")
    ax.text(0.012, 4 * np.exp(-2) * 1.15, r"$4e^{-2}$ (Theorem 1 optimum)", fontsize=6.6, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-5, 3)
    ax.set_xlabel(r"input pump intensity $I_0/I_s$"); ax.set_ylabel(r"$\eta$ / bound")
    ax.legend(frameon=False, loc="lower left", fontsize=6.2, markerscale=2.5); clean(ax)
    fig.savefig(os.path.join(outdir, "fig2_bound.pdf")); plt.close(fig)


def fig3():
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5), gridspec_kw=dict(wspace=0.42))
    for s in U["sat"]:
        c = COL[s["name"]]; p = np.array(s["p"])
        axs[0].plot(p, s["eta"], color=c, label=s["name"])
        axs[0].plot(p, s["ss"], color=c, lw=0.7, ls=":")
        if s["s_inf"] > 0:
            m = p > 30
            axs[0].plot(p[m], s["s_inf"] * p[m], color=c, lw=0.8, ls="--")
        axs[1].plot(p, s["film"], color=c, label=s["name"])
        pv = 15.0 * min(1.0, 1.0 / s["pars"]["r23"])
        for ax, yy in ((axs[0], s["eta"]), (axs[1], s["film"])):
            yv = np.exp(np.interp(np.log(pv), np.log(p), np.log(np.maximum(yy, 1e-12))))
            ax.plot([pv], [yv], marker="|", ms=9, mew=1.6, color=c)
        dm = [r for r in DM if r["tag"] == "res|" + s["name"]]
        axs[2].plot([r["p0"] for r in dm], [r["ratio"] for r in dm], "o-", ms=3.0, lw=0.9, mfc="none", mew=0.8, color=c)
        pvd = dm[0]["p_valid"]
        axs[2].plot([pvd], [np.interp(np.log(pvd), np.log([r["p0"] for r in dm]), [r["ratio"] for r in dm])], marker="|",
                    ms=9, mew=1.6, color=c)
    pb = np.array(U["sat"][0]["p"])
    axs[0].plot(pb, pb / 4, color=INK, lw=0.9, ls="-.", label="Theorem 2 ($r_{23}\\geq1$)")
    for p, mk in [(10.0, "s"), (100.0, "D")]:
        dm = [r for r in DM if r["tag"] == f"det|{p:g}"]
        axs[2].plot([p] * len(dm), [r["ratio"] for r in dm], mk, ms=3.6, color=PURPLE, mfc="none", mew=0.9,
                    label="detuned (rule), $|\\delta_{12}|=2$–$8\\gamma$" if p == 10.0 else None)
    axs[2].axhline(1, color=INK2, lw=0.6)
    axs[2].set_ylim(0.5, 1.1); axs[2].set_ylabel(r"$\eta_\mathrm{DM}/\eta_\mathrm{rate\ eq.}$")
    axs[2].legend(frameon=False, loc="lower left", fontsize=6.0)
    for ax in axs:
        ax.set_xscale("log"); ax.set_xlim(1e-2, 1e3)
        ax.set_xlabel(r"pump intensity $I/I_s$"); clean(ax)
    axs[0].set_yscale("log"); axs[1].set_yscale("log"); axs[2].set_xlim(5e-2, 5e2)
    axs[0].set_ylim(1e-3, 300); axs[0].set_ylabel(r"waveguide $\eta/\eta_s$ (optimum length)")
    axs[0].legend(frameon=False, loc="upper left", fontsize=5.8)
    axs[1].set_ylim(1e-3, 1); axs[1].set_ylabel(r"thin film $2\eta/(\eta_s A_\omega A_{2\omega})$", labelpad=1)
    panel_label(axs[0], "a"); panel_label(axs[1], "b"); panel_label(axs[2], "c")
    fig.savefig(os.path.join(outdir, "fig3_saturation.pdf")); plt.close(fig)


def fig4():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw=dict(wspace=0.3))
    ax = axs[0]
    for p, c in [(1.0, AQUA), (10.0, BLUE), (100.0, ORANGE)]:
        for b, ls in [(1e-4, "-"), (1e-2, "--")]:
            rr = sorted([r for r in D if r["name"] == "ADQW-like" and r["p"] == p and r["b"] == b], key=lambda r: r["d12"])
            ax.plot([r["d12"] for r in rr], [r["x"] for r in rr], color=c, ls=ls, lw=1.3 if ls == "-" else 1.0,
                    label=f"$I_0/I_s={p:g}$" if b == 1e-4 else None)
    ax.axhline(np.exp(-2), color=INK2, lw=0.7, ls=":"); ax.text(-7.8, np.exp(-2) * 1.12, "$e^{-2}$", fontsize=7, color=INK2)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 0.4); ax.set_xlim(-8, 8)
    ax.set_xlabel(r"pump detuning $\delta_{12}/\gamma$ ($\delta_{13}$ set by $u=1$)"); ax.set_ylabel(r"$\eta/(\eta_s I_0/I_s)$")
    ax.legend(frameon=False, loc="lower center", fontsize=6.4, ncol=3); clean(ax); panel_label(ax, "a")
    ax.text(0.98, 0.97, "solid: $\\alpha_{bg}=10^{-4}A_{12}$\ndashed: $10^{-2}A_{12}$", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.3, color=INK2)
    ax = axs[1]
    for d, c, lab in [("ad", BLUE, "ADQW"), ("tw", ORANGE, "TW")]:
        for P, ls in [(1.0, ":"), (4.0, "-"), (10.0, "--")]:
            rr = sorted([r for r in H if r["design"] == d and r["P"] == P], key=lambda r: r["dhw"])
            ax.plot([r["dhw"] * 1e3 for r in rr], [r["eta"] * 100 for r in rr], color=c, ls=ls, lw=1.3,
                    label=f"{lab}, {P:g} W")
    ax.set_xlabel(r"$\hbar\omega-124$ meV"); ax.set_ylabel(r"$\eta$ (%), qwkit, phase matched")
    sec = ax.secondary_xaxis("top", functions=(lambda x: 1239.84 / (124.0 + np.asarray(x)), lambda l: 1239.84 / np.asarray(l) - 124.0))
    sec.set_xticks([20, 16, 14, 12, 10, 9]); sec.set_xlabel(r"pump wavelength ($\mu$m)", fontsize=7.5)
    sec.tick_params(labelsize=7)
    ax.set_yscale("log"); ax.legend(frameon=False, fontsize=6.0, ncol=2, loc="lower center"); ax.set_ylim(0.03, 10)
    clean(ax); panel_label(ax, "b")
    fig.savefig(os.path.join(outdir, "fig4_detuning.pdf")); plt.close(fig)


def fig5():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw=dict(wspace=0.3))
    for ax, d, lab in [(axs[0], "ad", "a"), (axs[1], "tw", "b")]:
        for ns, ls, mk in [("5e+11", "-", "o"), ("1e+11", "--", "s")]:
            key = f"{d}_{ns}"
            sim = sorted([r for r in V["prop"] if r["design"] == d and f"{r['ns']:.0e}" == ns], key=lambda r: r["P"])
            mod = V["model"][key]
            P = np.array([r["P"] for r in sim])
            c = BLUE if d == "ad" else ORANGE
            ax.plot(P, [r["eta"] * 100 for r in sim], mk, color=c, ms=3.2, mfc="none" if ns == "1e+11" else c, mew=0.8,
                    label=f"qwkit, $n_s$={ns.replace('e+11','')}$\\times10^{{11}}$ cm$^{{-2}}$")
            ax.plot(P, [r["eta"] * 100 for r in mod], color=c, ls=ls, lw=1.1, label="universal ladder" if ns == "5e+11" else None)
            if ns == "5e+11":
                ax.plot(P, [r["eta_ss"] * 100 for r in mod], color=INK2, ls=":", lw=1.0, label="Theorem 1 (unsaturated)")
                ax.plot(P, [r["bound"] * 100 for r in mod], color=INK, ls="-.", lw=1.0, label="Theorem 2 bound")
                ax.axvline(N[key]["P_valid_W"], color=MUTED, lw=0.8)
                ax.text(N[key]["P_valid_W"] * 1.1, 0.0015, r"$\hbar\Omega=|D|$", fontsize=6.5, color=INK2, rotation=90)
        if d == "tw":
            for f, c2 in [(3.0, PURPLE), (10.0, AQUA)]:
                rr = sorted([r for r in B if r["f"] == f], key=lambda r: r["P"])
                ax.plot([r["P"] for r in rr], [r["eta"] * 100 for r in rr], color=c2, lw=1.2,
                        label=f"qwkit, bottleneck {rr[0]['tau1_ps']:.2f} ps")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-3, 100); ax.set_xlim(8e-3, 400)
        ax.set_xlabel("pump power (W)"); ax.set_ylabel(r"$\eta$ (%), phase matched")
        ax.set_title("ADQW" if d == "ad" else "Triple well", color=INK)
        ax.legend(frameon=False, fontsize=5.9, loc="upper left"); clean(ax); panel_label(ax, lab)
    fig.savefig(os.path.join(outdir, "fig5_qwkit.pdf")); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("figures written to", os.path.abspath(outdir))
