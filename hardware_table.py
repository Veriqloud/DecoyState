#!/usr/bin/env python3
"""
============================================================
  STANDALONE Hardware Calibration Table Generator
  AUREA SPD System — Performance Reference
============================================================

This tool uses the FULL Rusca et al. 2018 security bounds (same as main simulator)
to generate accurate hardware calibration tables.

Usage:
    python hardware_table.py params_aurea.json

Output: 
    - Console table with all parameters
    - PNG figure: hardware_calibration_[system].png
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  Helper functions (from v13)
# ============================================================

def hbin(p):
    p = np.clip(p, 1e-15, 1-1e-15)
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def delta(n, eps):
    return np.sqrt(n/2 * np.log(1/eps)) if n > 0 else 0.0

def compute_nsifted(nPP):
    PE_COEFF = 70
    return int((2*nPP + PE_COEFF**2 + np.sqrt((2*nPP+PE_COEFF**2)**2 - 4*nPP**2))/2)

def min_non_neg_skr(x):
    if x < 0.0:        return 0.0
    elif 0.0 < x < 1.0: return 0.0
    else:              return x

# ============================================================
#  Afterpulse functions
# ============================================================


def compute_pap(A, tau, R, dead_time_us, mu, eta, Pdc, coeff):
    T=80/R
    Pc0 = 1-np.exp(-mu*eta) + Pdc
    Pnc = 1 - coeff*Pc0
    dt1=T*dead_time_us # Deadtime in R*pulse_distance us

    u1 = 1/ (1/tau[0] - R*np.log(Pnc))
    u2 = 1/ (1/tau[1] - R*np.log(Pnc))
    u3 = 1/ (1/tau[2] - R*np.log(Pnc))

    R1=R*(1-dt1)
    
    pap1 = Pnc**R1*A[0] * np.exp(-(dt1+1)/u1) / (1 - np.exp(-1/u1)) 
    pap2 = Pnc**R1*A[1] * np.exp(-(dt1+1)/u2) / (1 - np.exp(-1/u2)) 
    pap3 = Pnc**R1*A[2] * np.exp(-(dt1+1)/u3) / (1 - np.exp(-1/u3)) 

    return (pap1 + pap2 + pap3)



# ============================================================
#  Full Rusca compute_all (from v13)
# ============================================================

def compute_all(d_km, e_det, p1, pZ_val, mu1_val, mu2_val, cfg):
    """Full Rusca et al. 2018 security bounds calculation."""
    
    # Unpack config
    alpha = cfg['alpha']
    odr_losses = cfg['odr_losses']
    eta_bob = cfg['eta_bob']
    pdc = cfg['pdc']
    f_rep = cfg['f_rep']
    dead_us = cfg['dead_us']
    nZ = cfg['nZ']
    K = cfg['K']
    esec = cfg['esec']
    ecor = cfg.get('ecor', esec)
    fEC = cfg['fEC']
    coeff = cfg.get('coeff', 1.0)
    Protocol_symmetric = cfg.get('Protocol_symmetric', False)
    afterpulse_cfg = cfg.get('afterpulse', {})
    if afterpulse_cfg :
        A = afterpulse_cfg.get('A') 
        tau = afterpulse_cfg.get('tau')
        R = afterpulse_cfg.get('R')
    else: 
        A=[0,0,0]
        tau=[1,1,1]
        R=1  
    
    p2 = 1 - p1
    eps1 = esec / K
    
    # Sanity checks
    if mu1_val <= mu2_val + 0.01:  return None
    if mu2_val <= 0 or mu2_val >= mu1_val:  return None
    if p1 <= 0 or p2 <= 0 or p1 >= 1 or pZ_val <= 0 or pZ_val >= 1:  return None
    
    t0_l = p1*np.exp(-mu1_val) + p2*np.exp(-mu2_val)
    t1_l = p1*np.exp(-mu1_val)*mu1_val + p2*np.exp(-mu2_val)*mu2_val
    
    eta = 10**(-(alpha*d_km+odr_losses)/10) * eta_bob
    
    # Detection probabilities with afterpulse
    D1 = 1 - np.exp(-mu1_val*eta) + pdc
    D2 = 1 - np.exp(-mu2_val*eta) + pdc
    p_ap1 = compute_pap(A, tau, R, dead_us, mu1_val, eta, pdc, coeff)
    p_ap2 = compute_pap(A, tau, R, dead_us, mu2_val, eta, pdc, coeff)
    R1 = D1 * (1 + p_ap1)
    R2 = D2 * (1 + p_ap2)
    Pdt = p1*R1 + p2*R2
    if Pdt <= 0:  return None
    
    cdt = coeff/(1 + f_rep*Pdt*dead_us*1e-6)
    
    if Protocol_symmetric:
        nS = compute_nsifted(nZ)
        nX = nS - nZ
        Ntot = nS / (cdt * 0.5 * Pdt)
    else:
        pX_val = 1.0 - pZ_val
        nX = nZ * (pX_val/pZ_val)**2
        denom = cdt * pZ_val**2 * Pdt
        if denom <= 0.0:  return None
        Ntot = nZ / denom
    
    nZ1 = nZ * p1*R1/Pdt;  nZ2 = nZ * p2*R2/Pdt
    nX1 = nX * p1*R1/Pdt;  nX2 = nX * p2*R2/Pdt
    
    dnZ = delta(nZ, eps1); dnX = delta(nX, eps1)
    if dnZ >= nZ1 or dnZ >= nZ2:  return None
    
    # QBER with afterpulse
    Pdt1 = coeff*(1 - np.exp(-mu1_val*eta) + pdc) * (1+coeff*p_ap1)
    E1 = (coeff*(1-np.exp(-mu1_val*eta))*(e_det +coeff*p_ap1/2) + coeff*pdc*(1+coeff*p_ap1)/2) / Pdt1
    Pdt2 = coeff*(1 - np.exp(-mu2_val*eta) + pdc) * (1+coeff*p_ap2)
    E2 = (coeff*(1-np.exp(-mu2_val*eta))*(e_det +coeff*p_ap2/2) + coeff*pdc*(1+coeff*p_ap2)/2) / Pdt2
    mZ1 = nZ1*E1; mZ2 = nZ2*E2; mZ = mZ1+mZ2; eobs = mZ/nZ
    mX1 = nX1*E1; mX2 = nX2*E2; mX = mX1+mX2
    
    dmX = delta(mX, eps1); dmZ = delta(mZ, eps1)
    
    # Weighted Hoeffding counts
    nZ1pw = (np.exp(mu1_val)/p1)*(nZ1+dnZ)
    nZ2mw = (np.exp(mu2_val)/p2)*(nZ2-dnZ)
    nX1pw = (np.exp(mu1_val)/p1)*(nX1+dnX)
    nX2mw = (np.exp(mu2_val)/p2)*(nX2-dnX)
    mX1pw = (np.exp(mu1_val)/p1)*(mX1+dmX)
    mX2mw = (np.exp(mu2_val)/p2)*(mX2-dmX)
    
    # Rusca bounds
    sz0u = 2*((t0_l*np.exp(mu2_val)/p2)*(mZ2+dmZ) + dnZ)
    sz0l_raw = (t0_l/(mu1_val-mu2_val))*(mu1_val*nZ2mw - mu2_val*nZ1pw)
    sz0l = max(sz0l_raw, 0.0)
    
    pref = (t1_l*mu1_val)/(mu2_val*(mu1_val-mu2_val))
    term_d = nZ2mw
    term_s = -(mu2_val/mu1_val)**2 * nZ1pw
    term_v = -(mu1_val**2-mu2_val**2)/mu1_val**2 * (sz0u/t0_l)
    sz1l = max(pref*(term_d+term_s+term_v), 0.0)
    
    sz0uX = 2*((t0_l*np.exp(mu2_val)/p2)*(mX2+dmX) + dnX)
    sx1l = max(pref*(nX2mw-(mu2_val/mu1_val)**2*nX1pw
                      -(mu1_val**2-mu2_val**2)/mu1_val**2*(sz0uX/t0_l)), 0.0)
    vx1u = max((t1_l/(mu1_val-mu2_val))*(mX1pw-mX2mw), 0.0)
    
    phi_raw = min(vx1u/sx1l, 0.5) if sx1l > 0 else 0.5
    
    # Rusca gamma smoothing
    if 0 < phi_raw < 0.5 and sz1l > 0 and sx1l > 0:
        arg = max(((sz1l+sx1l)/(sz1l*sx1l*(1-phi_raw)*phi_raw))
                  * (K**2/esec**2), 1.0)
        gam = np.sqrt((sz1l+sx1l)*(1-phi_raw)*phi_raw
                      /(sz1l*sx1l*np.log(2))*np.log2(arg))
    else:
        gam = 0.0
    phi = min(phi_raw+gam, 0.5)
    
    # Key length
    overhead = 6*np.log2(K/esec) + np.log2(2/ecor)
    lEC = fEC * hbin(eobs) * nZ
    ell = max(sz0l + sz1l*(1-hbin(phi)) - lEC - overhead, 0.0)
    
    skr = min_non_neg_skr(ell * f_rep / Ntot) if Ntot > 0 else 0.0
    
    if np.isnan(ell) or np.isinf(ell) or ell < 0:  ell = 0.0
    if np.isnan(skr) or np.isinf(skr) or skr < 0:  skr = 0.0
    
    return {'skr': skr, 'eobs': eobs, 'phi': phi, 'count_rate': cdt*Pdt*f_rep}

# ============================================================
#  Optimization
# ============================================================

def optimize_params(d_km, edet, cfg):
    """Find optimal (μ₁, μ₂, p_μ₁, p_Z, dead_time) using full Rusca bounds."""
    
    # Grid from v13
    mu1_scan = np.linspace(0.06, 0.6, 10)
    mu2_frac = np.linspace(0.1, 0.45, 6)
    pm1_scan = np.linspace(0.17, 0.81, 12)
    #pZ_scan = np.arange(0.70, 0.96, 0.035)
    pZ_scan = np.linspace(0.35, 0.95, 10)

    # Dead time sweep (from Fig 5b)
    dead_scan = [4, 6, 10, 20, 40, 60, 80, 100]  # μs

    
    
    best_skr = 0
    best = None
    
         
    # Inner loops: protocol parameters
    for mu1_v in mu1_scan:
        #for mu2_v in mu2_scan:
        for frac in mu2_frac:
            mu2_v = frac * mu1_v
            if mu1_v <= mu2_v:
                continue
            # Outer loop: dead time
            for dead_us in dead_scan:
                # Make temporary config with this dead time
                cfg_temp = cfg.copy()
                cfg_temp['dead_us'] = dead_us
                
                for pm1_v in pm1_scan:
                    for pZ_v in pZ_scan:
                        
                        r = compute_all(d_km, edet, pm1_v, pZ_v, mu1_v, mu2_v, cfg_temp)
                        
                        if r is not None and r['skr'] > best_skr:
                            best_skr = r['skr']
                            best = {
                                'mu1': mu1_v,
                                'mu2': mu2_v,
                                'pm1': pm1_v,
                                'pZ': pZ_v,
                                'dead_us': dead_us,
                                'skr': r['skr'], 
                                'eobs': r['eobs'],
                                'count_rate': r['count_rate']
                            }
    
    return best

# ============================================================
#  Figure generation
# ============================================================

def save_table_figure(results_dict, label, cfg):
    """Save table as PNG figure."""
    
    # Create figure with more height for header/footer
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor('#FAFAFA')
    
    # Create axes for table in center, leaving room for header/footer
    ax = fig.add_subplot(111)
    ax.set_position([0.05, 0.08, 0.90, 0.80])  # [left, bottom, width, height]
    ax.axis('off')
    
    # Title at top
    fig.text(0.5, 0.96, f"Hardware Calibration Table — {label}",
             ha='center', va='top', fontsize=14, fontweight='bold', color='#1f4788')

    if cfg['Protocol_symmetric']:
        Protocol_running = "Protocol Symmetric"
    else:
        Protocol_running = "Protocol Asymmetric"
    
    # System info below title
    fig.text(0.5, 0.93, 
             #f"Dead time: {cfg['dead_us']:.1f} μs  |  "
             f"n_Z={cfg['nZ']:.0e}  |  "
             f"ε_sec={cfg['esec']:.0e}  |  "
             f"f_EC={cfg['fEC']:.2f}  |  "
             f"using {Protocol_running}",
             ha='center', va='top', fontsize=10, color='#555')
    
    # Column headers
    col_labels = ['e_det\n(%)', 'Distance\n(km)', 'SKR\n(bits/s)', 
                  'μ₁', 'μ₂', 'p_μ₁', 'p_Z', 'Dead\n(μs)', 'μ₂/μ₁', 'eobs', 'count_rate']
    
    # Build table data
    table_data = []
    row_colors = []
    
    for i, (edet, rows) in enumerate(results_dict.items()):
        for j, row_data in enumerate(rows):
            if j == 0:
                table_data.append([f"{edet*100:.2f}", row_data[0], row_data[1],
                                  row_data[2], row_data[3],
                                  row_data[4], row_data[5], row_data[6], row_data[7], row_data[8], row_data[9]])
            else:
                table_data.append(['', row_data[0], row_data[1], row_data[2],
                                  row_data[3], row_data[4],
                                  row_data[5], row_data[6], row_data[7], row_data[8], row_data[9]])
            
            row_colors.append('#E6EEF7' if i % 2 == 0 else 'white')
    
    # Create table
    tb = ax.table(cellText=table_data, colLabels=col_labels,
                  loc='center', cellLoc='center')
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1.0, 1.6)
    
    # Style header
    for j in range(len(col_labels)):
        cell = tb[0, j]
        cell.set_facecolor('#1f4788')
        cell.set_text_props(color='white', fontweight='bold', fontsize=9)
    
    # Style rows
    for i in range(len(table_data)):
        for j in range(len(col_labels)):
            cell = tb[i + 1, j]
            cell.set_facecolor(row_colors[i])
            
            if j == 0 and table_data[i][0] != '':
                cell.set_text_props(fontweight='bold', fontsize=9)
            
            if j == 2 and table_data[i][2] == '<0.1':
                cell.set_text_props(color='#d32f2f', fontweight='bold')
    
    # Footer notes at bottom
    notes_line1 = "Hardware Setup Instructions:"
    notes_line2 = "(1) Set Alice's laser attenuators to achieve μ₁ and μ₂  " \
                  "(2) Configure RNG: intensity probability p_μ₁, basis probability p_Z"
    notes_line3 = "(3) Set Bob's detector dead time  " \
                  "(4) Run protocol for n_Z pulses. Expected SKR shown at optimized parameters."
    
    fig.text(0.5, 0.05, notes_line1,
             ha='center', va='bottom', fontsize=9, 
             color='#1f4788', fontweight='bold')
    fig.text(0.5, 0.035, notes_line2,
             ha='center', va='bottom', fontsize=8, 
             color='#555', style='italic')
    fig.text(0.5, 0.02, notes_line3,
             ha='center', va='bottom', fontsize=8, 
             color='#555', style='italic')
    
    # Save
    safe_label = label.replace(' ', '_').replace('—', '').replace('/', '').replace(',', '')
    filename = f"hardware_calibration_{safe_label}.png"
    fig.savefig(filename, dpi=200, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"\nFigure saved: {filename}")
    plt.close(fig)

# ============================================================
#  Table generator
# ============================================================

def generate_table(config_file):
    """Generate hardware calibration tables."""
    
    with open(config_file) as f:
        cfg = json.load(f)
    
    label = cfg['label']
    dead_us_config = cfg['dead_us']
    
    # Ranges
    #distances = [0, 10, 25, 50, 75, 100]
    #edet_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    #distances = [0,18,43]
    distances = [0,18,43]
    edet_range = np.arange(1,14)/200 
    
    
    # Header
    print("=" * 130)
    print(f"  {label} — Hardware Calibration Reference")
    print("=" * 130)
    print(f"Configuration:")
    print(f"  - Fiber loss: {cfg['alpha']:.3f} dB/km")
    print(f"  - Bob efficiency: {cfg['eta_bob']*100:.0f}%")
    print(f"  - Repetition rate: {cfg['f_rep']/1e6:.1f} MHz")
    #print(f"  - Afterpulse: A={A:.4f}, τ={tau:.2f} μs")
    print(f"  - Security: n_Z={cfg['nZ']:.0e}, ε_sec={cfg['esec']:.0e}, f_EC={cfg['fEC']:.2f}")
    if cfg['Protocol_symmetric']:
        print(f"  - Protocol: Symmetric")
    else:
        print(f"  - Protocol: Asymmetric")
    print(f"  - Using FULL Rusca et al. 2018 bounds (same as main simulator)")
    print(f"  - Optimizing over (μ₁, μ₂, p_μ₁, p_Z, dead_time)")
    print("=" * 130)
    print()
    
    # Collect results
    results_dict = {}
    
    # Generate tables
    for edet in edet_range:
        print("─" * 130)
        #print(f"  Optical Misalignment: e_det = {edet*100:.2f}%")
        #print("─" * 130)
        print(f"{'edet':>5s} {'Distance':>10s}  {'SKR (bits/s)':>14s}  {'μ₁':>8s}  {'μ₂':>8s}  "
              f"{'p_μ₁':>8s}  {'p_Z':>8s}  {'Dead (μs)':>11s} {'μ₂/μ₁':>8s} {'eobs':>8s} {'count_rate':>14s}")
        print("-" * 130)
        
        results_dict[edet] = []
        
        for d in distances:
            result = optimize_params(d, edet, cfg)
            
            if result and result['skr'] >= 0.1:
                mu2_mu1 = result['mu2'] / result['mu1']
                dead_opt = result['dead_us']
                eobs = result['eobs']
                count_rate = result['count_rate']
                print(f"{edet*100:5.2f}% {d:6.0f} km  {result['skr']:13.1f}  "
                      f"{result['mu1']:9.3f}  {result['mu2']:9.3f}  "
                      f"{result['pm1']:7.2f}  "
                      f"{result['pZ']:8.2f}  {dead_opt:10.1f} {mu2_mu1:9.2f}"
                      f"{eobs*100:9.2f} {count_rate:14.1f}")
                
                results_dict[edet].append([
                    f"{d:.0f}", f"{result['skr']:.0f}",
                    f"{result['mu1']:.3f}", f"{result['mu2']:.3f}",
                    f"{result['pm1']:.2f}",
                    f"{result['pZ']:.2f}", f"{dead_opt:.1f}", f"{mu2_mu1:.2f}", f"{eobs*100:.2f}", f"{count_rate:.1f}"
                ])
            else:
                print(f"{edet*100:5.2f}% {d:6.0f} km  {'<0.1':>14s}  {'—':>8s}  {'—':>8s}  "
                      f"{'—':>8s}  {'—':>8s}  {'—':>11s} {'—':>8s} {'—':>8s} {'—':>14s}")
                
                results_dict[edet].append([
                    f"{d:.0f}", '<0.1', '—', '—', '—', '—', '—', '—', '—', '—', '—' 
                ])
        
        print()
    
    print("=" * 130)
    print("Hardware Setup Instructions:")
    print("=" * 130)
    print()
    print("1. ALICE'S LASER (Intensity Modulation):")
    print("   - Set attenuator #1 to achieve μ₁ (signal intensity)")
    print("   - Set attenuator #2 to achieve μ₂ (decoy intensity)")
    print("   - Verify μ₂/μ₁ ratio matches table")
    print()
    print("2. ALICE'S RANDOM NUMBER GENERATOR:")
    print("   - Intensity: send μ₁ with probability p_μ₁, else μ₂")
    print("   - Basis: use Z with probability p_Z, else X")
    print()
    print("3. BOB'S DETECTOR:")
    print("   - Set dead time to OPTIMIZED value shown in table (varies per point)")
    print()
    print("4. PROTOCOL:")
    print(f"   - Run for n_Z = {cfg['nZ']:.0e} pulses")
    print(f"   - Expected SKR shown at f_rep = {cfg['f_rep']:.0e} Hz")
    print()
    print("=" * 130)
    print("Notes:")
    print("  - '<0.1' means protocol fails (SKR < 0.1 bits/s)")
    print("  - Dead time is OPTIMIZED per (distance, e_det) point")
    print("  - Values use FULL Rusca finite-key bounds (same as main simulator)")
    print("  - For detailed figures and analysis, run qkd_1decoy_analysis_v13_v2.py")
    print("=" * 130)
    
    # Generate figure
    save_table_figure(results_dict, label, cfg)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python hardware_table.py params_aurea.json")
        sys.exit(1)
    
    generate_table(sys.argv[1])
