import numpy as np


def hbin(p):
    p = np.clip(p, 1e-15, 1-1e-15)
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def delta(n, eps):
    return np.sqrt(n/2 * np.log(1/eps)) if n > 0 else 0.0

#PE_COEFF = 70
# def compute_nsifted(nPP):
#     return int((2*nPP + PE_COEFF**2 + np.sqrt((2*nPP+PE_COEFF**2)**2 - 4*nPP**2))/2)

# def min_non_neg_skr(x):
#     if x < 0.0:        return 0.0
#     elif 0.0 < x < 1.0: return 0.0
#     else:              return x


def compute_final_length(nX, nZ, mX1, mX2, nX1, nX2, nZ1, nZ2, esec,K,nX,nZ, mu1_val, mu2_val,p1):
    """Full Rusca et al. 2018 security bounds calculation."""
    
    p2 = 1 - p1
    eps1 = esec / K

    t0_l = p1*np.exp(-mu1_val) + p2*np.exp(-mu2_val)
    t1_l = p1*np.exp(-mu1_val)*mu1_val + p2*np.exp(-mu2_val)*mu2_val

    dnZ = delta(nZ, eps1); dnX = delta(nX, eps1)
    if dnZ >= nZ1 or dnZ >= nZ2:  return None

    #mX1 = nX1*E1; mX2 = nX2*E2; mX = mX1+mX2
    
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
    ell = max(sz0l + sz1l*(1-hbin(phi)) - lEC - overhead, 0.0)
    
   
     
    return {'skr': skr, 'eobs': eobs, 'phi': phi, 'count_rate': cdt*Pdt*f_rep}


if __name__ == "__main__":


    #lEC = fEC * hbin(eobs) * nZ
    #skr = min_non_neg_skr(ell * f_rep / Ntot) if Ntot > 0 else 0.0