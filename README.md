# 1-Decoy State QKD — Security Bounds Analysis

Finite-key security analysis of the 1-decoy BB84 protocol following Rusca et al. (2018).  
Models two hardware systems: AUREA SPD detector and Rusca SNSPD reference.  
Application context: quantum token transmission over optical fibre.

---

## Usage

```bash
python3 qkd_1decoy_analysis.py params_aurea.json    # AUREA SPD system
python3 qkd_1decoy_analysis.py params_rusca.json    # Rusca 2018 SNSPD reference
```

---

## Config Files

Two config files are included in the repository:

- **`params_aurea.json`** — AUREA SPD system (our hardware): η_Bob=0.20, p_dc=6×10⁻⁷, f_rep=80 MHz, dead_time=10 μs, odr_losses=11.4 dB, nZ=10⁶
- **`params_rusca.json`** — Rusca 2018 SNSPD reference: η_Bob=0.50, p_dc=10⁻⁸, f_rep=1 GHz, dead_time=0.1 μs, odr_losses=0 dB, nZ=10⁷

All other parameters (μ₁, μ₂, pZ, pX, α, edet, εsec, εcor, fEC, K) are shared and defined in both files.  
Output figures are automatically named with the config label so files from different configs never overwrite each other.

---

## η_sys Convention

System efficiency follows Anne's explicit convention:

```
η_sys = η_Bob × 10^(-(att_channel + odr_losses) / 10)
```

where `odr_losses = 11.4 dB` for the AUREA system (internal optical losses),  
and `odr_losses = 0.0 dB` for the Rusca SNSPD reference.

At 25 km for AUREA: 5 dB (fibre) + 11.4 dB (odr) = **16.4 dB total** → η_sys ≈ 0.46%

---

## Papers

| References | |
|---|---|
| Rusca et al. (2018) *Appl. Phys. Lett.* **112**, 171104 | Main security proof — Appendix A (bounds) + B (SKR). Eqs. A16–A25, B8 |
| Lim, Curty, Walenta, Xu, Zbinden (2014) *Phys. Rev. A* **89**, 022307 | Finite-key 2-decoy. Eq. 3 defines weighted Hoeffding counts used throughout |
| Lo, Ma & Chen (2005) *PRL* **94**, 230504 | Asymptotic decoy-state theory — justifies μ₁ ≈ 0.5 asymptotically |
| Tomamichel, Lim, Gisin, Renner (2012) *Nature Comm.* **3**, 634 | Alternative μ phase error correction — equivalent to Rusca γ for our parameters |
| Fung, Ma & Chau (2010) *Phys. Rev. A* **81**, 012318 | Practical post-processing — phase error estimation |
| Zhao et al. (2005) *PRL* **96**, 070502 | Experimental decoy-state QKD reference |

---
