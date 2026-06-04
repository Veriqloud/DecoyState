import numpy as np
from scipy.optimize import curve_fit
import sys, os
import json
import qkd_1decoy_analysis_v13_compact as qc
import pickle
from matplotlib import pyplot as plt

def optimize_params(d_km, e_det, grids, p_ap_model=None, dead_us_in=None):
	"""Find (mu1*, mu2*, p_mu1*, p_Z*, dead_time*) maximising SKR at (d, e_det).
	
	Now includes dead_time optimization in the outer loop.
	If p_ap_model is provided (dict with 'A', 'tau', 'T_max_us'), then
	dead_time is optimized by sweeping [6, 10, 15, 20, 30, 40] μs.
	Otherwise uses fixed dead_us_in (or config default).
	
	Returns dict with keys mu1, mu2, pm1, pZ, dead_us, skr, skr_sp,
	or None if no valid optimum (SKR < 1 b/s everywhere on grid).
	"""
	mu1_scan = grids['mu1_scan']
	mu2_frac = grids['mu2_frac']
	pm1_scan = grids['pm1_scan']
	pZ_scan  = np.array([pZ]) if qc.Protocol_symmetric else grids['pZ_scan']
	dead_us_scan = grids['dead_us_scan']
	

	best_s = 0.0
	best = dict(mu1=np.nan, mu2=np.nan, pm1=np.nan, pZ=np.nan, 
				dead_us=np.nan, skr=0.0, skr_sp=0.0)

	if p_ap_model is True:
		dead_us_scann = dead_us_scan
	else:
		if dead_us_in:
			dead_us_scann = [dead_us_in]
		else:
			dead_us_scann = [qc.dead_us]		
	
	# Inner loops: protocol parameters
	for mu1_v in mu1_scan:
		for frac in mu2_frac:
			mu2_v = frac * mu1_v
			if mu1_v <= mu2_v:  continue
			for dead_val in dead_us_scann:
				if p_ap_model is False:
					p_ap_val = 0.0
				else:
					p_ap_val = compute_pap(p_ap_model["A"], p_ap_model["tau"], dead_val, p_ap_model["T_max_us"])
				for pm1_v in pm1_scan:
					for pZ_v in pZ_scan:
						r = qc.compute_all(d_km, e_det=e_det, p1=pm1_v,
										pZ_in=pZ_v, mu1_in=mu1_v, mu2_in=mu2_v,
										p_ap1=p_ap_val, p_ap2=p_ap_val, dead_us_in=dead_val)
						if r is not None and r['skr'] > best_s:
							best_s = r['skr']
							best = dict(mu1=mu1_v, mu2=mu2_v, pm1=pm1_v,
										pZ=pZ_v, dead_us=dead_val,
										skr=r['skr'], skr_sp=r['skr_sp'], eobs=r['eobs'])

	return best if best_s >= 1.0 else None



def compute_pap(A, tau_us, dead_time_us, T_max_us):
	"""Afterpulse probability integral."""
	if tau_us <= 0 or A <= 0:
		return 0.0
	return A * tau_us * (np.exp(-dead_time_us/tau_us) - np.exp(-T_max_us/tau_us))

def qber_dt(mu,eta,pap,pdc,F,e0):
	Pdt = F*(1 - np.exp(-mu*eta) + pdc) * (1+F*pap)
	return (F*(1-np.exp(-mu*eta))*(e0 + F*pap*0.5) + F*pdc*0.5*(1 + F*pap)) / Pdt


def fit_pap_from_qber(dead_times_us, qbers, T_max_us):
	"""Fit afterpulse (A, tau) from QBER calibration."""
	dt = np.array(dead_times_us, dtype=float)
	qb = np.array(qbers, dtype=float) / 100.0
	order = np.argsort(dt)
	dt, qb = dt[order], qb[order]
	
	e_det_base = qb[-1]
	delta_q = qb[:-1] - e_det_base
	dt_fit = dt[:-1]
	y_target = 2.0 * delta_q
	
	def model(dt_vals, A, tau):
		return A * tau * (np.exp(-dt_vals/tau) - np.exp(-T_max_us/tau))

	try:
		p0 = [0.1, 10.0]
		popt, _ = curve_fit(model, dt_fit, y_target, p0=p0,
							bounds=([0.0, 0.1], [10.0, 1000.0]), maxfev=5000)
		A, tau = popt
		return float(A), float(tau), float(e_det_base), True
	except:
		return 0.0, 10.0, float(e_det_base), False


# Tunable knobs for Fig 5 — change these to alter grid resolution / range
DEAD_SWEEP  = np.linspace(6, 100, 20)			# us — x-axis for panels a,b
EDET_FAMILY = [0.01, 0.02, 0.03, 0.04, 0.05]  # curves on panel b
# Coarsened grids for the (d x edet) cache
GRIDS_CACHE = dict(
	#mu1_scan = np.linspace(0.12, 1.0, 8),
	#mu2_frac = np.linspace(0.15, 0.85, 6),
	#pm1_scan = np.linspace(0.02, 0.90, 12),
	#pZ_scan  = np.arange(0.70, 0.96, 0.035),
	mu1_scan = np.linspace(0.06, 0.6, 10),
	mu2_frac = np.linspace(0.1, 0.45, 6),
	pm1_scan = np.linspace(0.17, 0.81, 12),
	pZ_scan  = np.linspace(0.35, 0.95, 10),
	dead_us_scan = np.array([4,6,10,20,40,70,100])
)


def _build_cache(d_km, dead_sweep_us, edet_family, p_ap_model,
					  grids=GRIDS_CACHE, force_recompute=False):
	"""Per-(edet, dead_time) optimisation for Fig 5 panel (b).

	p_ap_model is a callable: dead_time_us -> p_ap (dimensionless).
	Cache key: (edet, dead_time_us). Cache file includes (A, tau).
	"""
	total = len(edet_family) * len(dead_sweep_us)
	print(f"[fig cache miss] computing {len(edet_family)}x{len(dead_sweep_us)} = "
		  f"{total} optimisations at d={d_km:.0f} km...")
	cache_file = os.path.join(qc.save_dir,
		f'cache_{qc.safe_label}_A{p_ap_model['A']:.3f}_tau{p_ap_model['tau']:.1f}_d{d_km:.0f}.pkl')

	if os.path.exists(cache_file) and not force_recompute:
		try:
			with open(cache_file, 'rb') as f:
				cache = pickle.load(f)
			need = set((float(e), float(dt)) for e in edet_family
					   for dt in dead_sweep_us)
			if need.issubset(set(cache.keys())):
				print(f"[fig5 cache hit]  {os.path.basename(cache_file)}  "
					  f"({len(cache)} entries)")
				return cache
			else:
				print(f"[fig cache stale] missing {len(need - set(cache.keys()))} — "
					  "recomputing")
		except Exception as ex:
			print(f"[fig cache error] {ex} — recomputing")


	cache = {}
	i = 0
	for e in edet_family:
		for dt_us in dead_sweep_us:
			i += 1
			#p_ap_dt = p_ap_model(dt_us)
			# For Fig 5(b), we FORCE dead_time to this specific value
			# Pass the p_ap_model dict so p_ap is computed correctly
			r = optimize_params(d_km, e, grids, p_ap_model,
								dead_us_in=dt_us)  # Force this dead_time
			cache[(float(e), float(dt_us))] = r
			if r is not None and (i % 10 == 0 or i == total):
				print(f"  [{i:>3d}/{total}] edet={e*100:.0f}%, "
					  f"dead={dt_us:5.1f}us: "
					  f"SKR={r['skr']:.0f} "
					  f"eobs={r['eobs']:.4f}")
	with open(cache_file, 'wb') as f:
		pickle.dump(cache, f)
	print(f"[fig cache saved] {os.path.basename(cache_file)}")

	return cache


def build_fig(d_op,p_ap_model,t_bins,p_density, Optimize):
	"""Three-panel Fig 5: QBER vs dead_time, best SKR vs dead_time,
	model validation.

	p_ap_fitted: dict with keys A, tau, e_det_baseline, T_max_us
	ts_fit: dict from fit_pap_from_timestamps, or None
	"""
	
	# Closure: dead_time -> p_ap using the fitted A, tau
	#def p_ap_model(dt_us):
	#	return compute_pap(A_fit, tau_fit, dt_us, T_max_us)
 
	
	fig = plt.figure(figsize=(12, 9))
	fig.patch.set_facecolor('#FAFAFA')
	gs = qc.gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28,
						   left=0.07, right=0.97, top=0.93, bottom=0.06)

	fig.suptitle(
		rf"Fig — Afterpulse Analysis  —  {qc.label}  "
		rf"(d = {d_op:.0f} km, "
		rf"$T_{{\max}}$={T_max_us:.0f} μs, "
		+ ("Symmetric)" if qc.Protocol_symmetric else "Asymmetric)"),
		fontsize=12, fontweight='bold', color=qc.NAVY, y=0.975)

	# ── Panel (a): QBER vs dead_time (measured + fit) ──
	ax_a = fig.add_subplot(gs[0, 0])
	dt_meas = np.array(afterpulse_cfg['qber_calibration']['dead_time_us'],
					   dtype=float)
	qb_meas = np.array(afterpulse_cfg['qber_calibration']['qber_pct'],
					   dtype=float) / 100.0
	dt_fine = np.linspace(DEAD_SWEEP.min(), DEAD_SWEEP.max(), 200)
	pap = np.array(
		[compute_pap(p_ap_model['A'], p_ap_model['tau'], dt, p_ap_model['T_max_us']) for dt in dt_fine])
	qb_model = qc.edet + 0.5 * pap 
	qb_model2 = qber_dt(mu=0.5, eta=0.01, pap=pap, pdc=1.33e-6, F=0.7, e0=0.01)

	ax_a.plot(dt_fine, qb_model*100, '-', color=qc.BLUE, lw=2.0,
			  label=rf'Model: $e_{{det,0}}={qc.edet*100:.2f}\%$ + $p_{{ap}}/2$')
	ax_a.plot(dt_fine, qb_model2*100, '-', color=qc.NAVY, lw=2.0,
			  label=rf'Model: (F*(1-$\exp(-\mu\eta$))*(e0 + F*pap*0.5) + F*pdc*0.5*(1 + F*pap)) / Pdt')
	ax_a.plot(dt_meas, qb_meas*100, 'o', color=qc.RED, ms=10,
			  markeredgecolor='white', markeredgewidth=1.5,
			  label='Measured QBER', zorder=5)
	ax_a.axhline(qc.edet*100, color=qc.GREY, lw=0.8, ls=':',
				 label=rf'Baseline $e_{{det}} = {qc.edet*100:.2f}\%$')
	ax_a.legend(fontsize=8.5, loc='upper right')
	qc.style_axis(ax_a, title=r'(a)  QBER vs Dead Time  —  measurement + model fit',
			   xlabel='Dead time (μs)', ylabel='QBER (%)',
			   fontsize_title=10, fontsize_label=10)

	# ── Panel (c): QBER-calibrated model vs measured histogram ──
	ax_c = fig.add_subplot(gs[1, :])
	
	t_fine = np.linspace(0, 300, 500)
	
	# Measured histogram dots (if available)
	tc = t_bins
	pd = p_density
	mask = (tc >= 0) & (tc <= 300)
	# Convert to %/μs, subtract Poisson floor
	ax_c.plot(tc, pd, 'o', color=qc.GREY, ms=3, alpha=0.6,
			  label='Measured histogram', zorder=2)
	
	# QBER-calibrated model (used in security analysis)
	p_qber = p_ap_model['A'] * np.exp(-t_fine/p_ap_model['tau']) + 0.0005
	ax_c.plot(t_fine, p_qber, '-', color=qc.BLUE, lw=3.0,
			  label=rf'QBER-calibrated model',
			  zorder=5)
	
	ax_c.legend(fontsize=9, loc='upper right', framealpha=0.95)
	ax_c.set_xlim(0, 100)
	#ax_c.set_ylim(0, max(2.0, 1.1*p_qber[0]))
	ax_c.grid(True, alpha=0.2, ls=':')
	qc.style_axis(ax_c,
		title=r'(c)  QBER-calibrated afterpulse model validation',
		xlabel='Time after detection (μs)',
		ylabel=r'Afterpulse probability density (%/μs)',
		fontsize_title=10, fontsize_label=10)
	
	fig.savefig(os.path.join(qc.save_dir, f'fig5_afterpulse_{qc.safe_label}.png'),
				dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
	print(f"Figure saved: fig_afterpulse_{qc.safe_label}.png")
	

	if Optimize is False:
		return fig
	# ── Panel (b): best SKR vs dead_time, e_det family ──

	# Build per-(edet, dead_time) cache at operating distance
	fig_cache = _build_cache(d_op, DEAD_SWEEP, EDET_FAMILY, p_ap_model)

	ax_b = fig.add_subplot(gs[0, 1])
	cmap = plt.cm.viridis
	colors_edet = [cmap(i/(len(EDET_FAMILY)-1)) for i in range(len(EDET_FAMILY))]

	# Collect per-edet optima for table + star markers
	optima = []   # list of dicts: {edet, dt_opt, skr_opt, dt_current, skr_current}
	for e, col in zip(EDET_FAMILY, colors_edet):
		skr_arr = np.full(len(DEAD_SWEEP), np.nan)
		for i, dt_us in enumerate(DEAD_SWEEP):
			r = fig_cache.get((float(e), float(dt_us)))
			if r is not None:
				skr_arr[i] = r['skr']
				eobs = r['eobs']
		ax_b.semilogy(DEAD_SWEEP, skr_arr, 'o-', color=col, lw=1.8, ms=4,
					  label=rf'$e_{{det}}={e*100:.0f}\%$')

		# Per-curve optimum — argmax over valid points
		if np.any(~np.isnan(skr_arr)):
			idx_opt = int(np.nanargmax(skr_arr))
			dt_opt  = DEAD_SWEEP[idx_opt]
			skr_opt = skr_arr[idx_opt]
			# Star marker at optimum
			ax_b.plot(dt_opt, skr_opt, marker='*', ms=18, color=col,
					  markeredgecolor='black', markeredgewidth=0.8,
					  zorder=6, clip_on=False)
			# SKR at current dead_us config (nearest grid point)
			idx_curr = int(np.argmin(np.abs(DEAD_SWEEP - qc.dead_us)))
			skr_curr = skr_arr[idx_curr]
			optima.append(dict(edet=e, dt_opt=dt_opt, skr_opt=skr_opt,
							   dt_current=qc.dead_us,
							   skr_current=skr_curr if not np.isnan(skr_curr) else 0.0,
							   eobs=eobs))

	# Mark measured calibration dead times (light grey dotted)
	for dt_v in dt_meas:
		ax_b.axvline(dt_v, color=qc.GREY, lw=0.6, ls=':', alpha=0.4)

	# Mark current config dead_us (bold red dashed) — "you are here"
	ax_b.axvline(qc.dead_us, color=qc.RED, lw=1.5, ls='--', alpha=0.7, zorder=4,
				 label=rf'current $t_d={qc.dead_us:.0f}$ μs')

	ax_b.legend(fontsize=7.5, loc='lower right', ncol=2)
	ax_b.set_xlim(DEAD_SWEEP.min(), DEAD_SWEEP.max())
	qc.style_axis(ax_b,
		title=rf'(b)  Best-achievable SKR vs Dead Time  @ d={d_op:.0f} km  '
			  r'(★ = per-curve optimum)',
		xlabel='Dead time (μs)', ylabel='SKR (bits/s)',
		fontsize_title=10, fontsize_label=10)

	# Print optima table to terminal
	print(f"\n{'Optimal dead time per e_det (d = ' + str(int(d_op)) + ' km)':^70s}")
	print("-" * 70)
	print(f"{'e_det':>7s}  {'dt_opt (us)':>12s}  {'SKR_opt':>10s}  "
		  f"{'dt_curr (us)':>13s}  {'SKR_curr':>10s}  {'gain':>6s} {'eobs':>6s}")
	print("-" * 70)
	for o in optima:
		gain_pct = ((o['skr_opt'] / o['skr_current']) - 1) * 100 \
				   if o['skr_current'] > 0 else float('inf')
		gain_str = f"{gain_pct:+5.0f}%" if np.isfinite(gain_pct) else "  —  "
		print(f"  {o['edet']*100:4.0f}%  {o['dt_opt']:12.1f}  "
			  f"{o['skr_opt']:10.0f}  {o['dt_current']:13.1f}  "
			  f"{o['skr_current']:10.0f}  {gain_str} {o['eobs']:6.4f}")
	print("-" * 70)

	return fig

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Usage: python fit_afterpulse_with_qber params_calib.json --show_optimize")
		sys.exit(1)
	
	config_file = sys.argv[1]
	Optimize = False
	if len(sys.argv) > 2:
		if "--show_optimize" in sys.argv[2]:
			Optimize = True
			

	with open(config_file) as f:
		cfg = json.load(f)


	#Get afterpulse parameters
	afterpulse_cfg = cfg.get('afterpulse', {})
	T_max_us = 10000000 #afterpulse_cfg.get('T_max_us', 100.0)
	qber_cal = afterpulse_cfg.get('qber_calibration')


	if qber_cal:
		A, tau_us, e_base, ok = fit_pap_from_qber(
			qber_cal['dead_time_us'], qber_cal['qber_pct'], T_max_us)
	else:
		A, tau_us = 0.0, 0.0


	print(f"A = {A}")
	print(f"tau_us = {tau_us}")

	print(f"verif (we suppose mu = 0.01):")
	mu=0.5
	eta=0.01
	Pdc=1.33e-6
	F=0.7
	dead_times_us = qber_cal['dead_time_us']
	qbers = qber_cal['qber_pct']
	Adt = np.array(dead_times_us)
	Aqb = np.array(qbers) / 100.0
	for i in np.arange(len(Adt)):
		dead_time_us = Adt[i]
		pap = compute_pap(A, tau_us, dead_time_us, T_max_us)
		qber_verify = qber_dt(mu,eta,pap,Pdc,F,e0=0.01)
		print(f"qber_verify={round(qber_verify,4)} qber_measured={round(Aqb[i],4)}")

	# plot figs
	d_op	= cfg.get('d_operating_km', 25.0)
	p_ap_model = { 
		"A" : A,
		"tau": tau_us,
		"T_max_us" : T_max_us
		}

	ts_path = afterpulse_cfg.get('timestamp_file')
	#ts_full = os.path.join(save_dir, ts_path) \
	#		  if not os.path.isabs(ts_path) else ts_path

	data = np.loadtxt(ts_path)
	data = data[data != 0]
	dt_s = np.diff(data) / 80e6	   # inter-arrival times in seconds
	dt_us_arr = dt_s * 1e6

	# Histogram, normalised to probability density (1/us)
	bin_us = 1
	edges = np.arange(0, T_max_us + bin_us, bin_us) - 0.5
	h, b  = np.histogram(dt_us_arr, bins=edges)
	centers = (b[:-1] + b[1:]) / 2
	density = h / (h.sum() * bin_us)  # probability per us, integrates to 1
	t_bins=centers 
	p_density=density
	
	build_fig(d_op,p_ap_model,t_bins,p_density,Optimize)
	plt.show()
