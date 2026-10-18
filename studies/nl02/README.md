# NL-02: Rabi-frequency bound and saturation limits of resonant ISB SHG

Scripts behind `paper_nl02/main.tex` (`NL02_paper.pdf`). Run everything with `studies/rerun_nl02.sh`.

| Script | Output | Content |
| --- | --- | --- |
| `model3.py` | — | Dimensionless three-level ladder (rate equations, coupled modes, QPM), Theorem 1 `g(u)`, Theorem 2 bound |
| `dm_check.py` | `dm_check.json` | Independent full steady-state three-level density matrix (RWA) propagation vs the ladder |
| `dm_local.py` | `dm_local.json` | Same, with the resonant index compensated at the local intensity (chirped phase matching) |
| `study_universal.py` | `universal.json` | Theorem 1 (300 random ladders), Theorem 2 (2000 random saturated/QPM cases), saturation curves, thin film |
| `asym_check.py` | `asym_check.json` | Large-intensity asymptote, Eq. (asym) |
| `detuning_study.py` | `detuning.json` | Detuning rule (u = 1) |
| `qwkit_validation.py` | `qwkit_validation.json` | Full qwkit ADQW and triple well vs mapped ladder; identity scans |
| `bottleneck.py` | `bottleneck.json` | Triple well with faster intermediate-state decay |
| `hw_scan.py` | `hw_scan.json`, `hw_scan_wg.json` | Pump photon-energy scan, waveguide recomputed per wavelength |
| `collect_numbers.py` | `numbers.json` | Every number quoted in the manuscript, including the chi(2) phase drift and thin-film peaks (fine grid), P_s and I/I_s at the detuned wavelengths, the density matrix at the detuned operating points, and the depolarization/Hartree shifts with equalized ladder populations |
| `paper_figures.py` | `paper_nl02/figures/*.pdf` | Figures 1–5 |
