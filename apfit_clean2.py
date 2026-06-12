import numpy as np 
import numpy as np, matplotlib.pylab as plt
from scipy.optimize import curve_fit
from sympy import symbols, solve
import sys

Lqber = np.array([50, 15.4, 9.3, 4.2, 50, 8.9, 6.0, 2.8])/100
Lmu = [0.0, 0.01, 0.1, 1.0, 0.01, 0.1, 1.0]
Ldt = [4,4,4,4,10,10,10,10]
Rlist = [1,2,4,5,8,10,16,20,40,80]
Rlist = [16,20,40,80]
#Best value : 
Rlist=[1]

print(Lqber)

print("With paramters :")
f_rep = 80e6
minimum_attenuation = 20
dkm = 0
eta = 10**(-(minimum_attenuation + dkm *0.23)/10)
F = 0.78 #coeff gate size
Pdc = 1.33e-6

print(f"eta = {eta}")
print(f"F = {F}")
print(f"Pdc = {Pdc}")



	
def pnextclick(n, mu, eta, nd, a1, tau1, a2, tau2, pdc):
	f = 0.26
	F = 0.78
	pc = 1-np.exp(-mu*eta) + pdc
	pnc = 1- pc
	pap = a1* np.exp(-n * (1 / tau1)) + a2 * np.exp(-n * (1 / tau2))
	B1 = a1/(1-np.exp(-1/tau1))
	B2 = a2/(1-np.exp(-1/tau2))
	L1 = B1*np.exp(-(nd + 1)/tau1)*(1-np.exp(-(n - nd - 1) / tau1))
	L2 = B2*np.exp(-(nd + 1)/tau2)*(1-np.exp(-(n - nd - 1) / tau2))
	return pnc**(n-nd-1) * (1 - L1 - L2) * (pap + pc)

def model(n, a1, tau1, a2, tau2):
	nd = DEADTIME
	pc = 1-np.exp(-MU*eta) + Pdc
	pnc = 1- pc
	pap = a1* np.exp(-n * (1 / tau1)) + a2 * np.exp(-n * (1 / tau2))
	B1 = a1/(1-np.exp(-1/tau1))
	B2 = a2/(1-np.exp(-1/tau2))
	L1 = B1*np.exp(-(nd + 1)/tau1)*(1-np.exp(-(n - nd - 1) / tau1))
	L2 = B2*np.exp(-(nd + 1)/tau2)*(1-np.exp(-(n - nd - 1) / tau2))
	return pnc**(n-nd-1) * (1 - L1 - L2) * (pap + pc)



def pap_sum_inf(dt, A1, u1, A2, u2, c):
	pap1 = A1 * np.exp(-(dt+2)/u1) / (1 - np.exp(-1/u1))
	pap2 = A2 * np.exp(-(dt+2)/u2) / (1 - np.exp(-1/u2))
	pdc = c 
	return (pap1 + pap2), pdc

def compute_qber(dt, mu, eta, a1, tau1, a2, tau2, pdc, e0,f):
	pap0 = compute_pap(dt, mu, eta, a1, tau1, a2, tau2)
	print("1+pap = ", 1+pap0)
	pap_inf = 1 / (1-pap0) # sum_inf = pap0^n 
	print("pour info pap_inf = ", pap_inf)
	#pap_inf2 = pap0 / (1-pap0) # = sum_inf -1
	pq = 1 - np.exp(-mu*eta)
	cdt = 1 / (1 + (pq + Pdc)*(1+pap0)*f_rep*DT*1e-6)
	pdt = pq + f*pdc + f*pap0*(pdc+pq)
	return (pq*e0 + f*(pdc + pap0*(pdc+pq))*0.5)/pdt


def compute_pap(dt, mu, eta, a1, tau1, a2, tau2):
	Pc = 1-np.exp(-mu*eta) + Pdc
	pnc = 1 - Pc
	ndt1=T*dt # Deadtime for divided Unit of time
	   
	u1 = 1/ (1/tau1 - R*np.log(pnc))
	u2 = 1/ (1/tau2 - R*np.log(pnc))

	pnap = 1 - a1*np.exp(-(ndt1 + 1)/tau1) - a2*np.exp(-(ndt1 + 1)/tau2)

	pap1 = a1 * np.exp(-(ndt1+2)/u1) / (1 - np.exp(-1/u1)) 
	pap2 = a2 * np.exp(-(ndt1+2)/u2) / (1 - np.exp(-1/u2)) 

	return pnap*(pap1 + pap2)



#filename = ["dt10_mu0.01.txt"]
#labelname = [" dt"] 

#filename = sys.argv[1]

#"dt4_mu0.txt"/"dt10_mu1.txt" : can t fit
#"dt4_mu1.0.txt","dt4_mu0.1.txt", "dt10_mu0.txt",   "dt10_mu0.1.txt"
#"dt10_mu0.01.txt"

#best value only
#for filename in ["wide_gates.txt"]:
#for filename in ["dt4_mu0.01.txt", "wide_gates.txt"]:	
for filename in [sys.argv[1]]:
	print("")
	print(f"Filename = {filename}")

	if "dt10_" in filename:
		DT = 10 # UT
	elif "dt4_" in filename:
		DT = 4 # UT
	elif filename == "wide_gates.txt":
		DT = 4
		DEADTIME = 335
		#p0 = [0.9, 1, 0.1, 10]
	else:
		print("no known DT")
		sys.exit()


	if "mu0.t" in filename:
		MU = 0 
	elif "mu0.01" in filename:
		MU = 0.01
	elif "mu0.1" in filename:
		MU = 0.1
	elif "mu1." in filename:
		MU = 1.0
	elif filename == "wide_gates.txt":
		MU = 0.005
	else:
		print("no known MU")
		sys.exit()


	print("DT = ", DT)
	print("MU = ", MU)


	data = np.loadtxt(filename)
	data = data[data!=0]
	dt = data[1:] - data[:-1]

	Nsent = data[-1]
	print(f"Nsent : ", Nsent)
	Nrecv = len(data)
	print(f"total count", Nrecv)
	pdet1= Nrecv/Nsent
	if MU != 0: 
		eta_deduced = pdet1 / ((F - pdet1*f_rep*DT*1e-6)*MU)
		print(f"If mu = 1 (ie negl DC and AP), eta_deduced = {eta_deduced}, must be >= eta_ref = {eta}")

	#if MU == 0.0:
	print(f"Estimate DC for mu=0:")
	print("  DC <= ", pdet1)
	Pdc_ref = pdet1 /2
	print(f"  Pdc_ref = {Pdc_ref} (Attention, F and DT is incuded, otherwise Pdc_ref_woF = {Pdc_ref/F})")
	print(f"  DC with Pdc_ref {Pdc_ref} = ", Pdc_ref*Nsent)



	
	


	for R in Rlist:
		T = 80/R # division of UT, ie a unit is 1/T microsecond
		print(f"R = {R}")


		#dt = dt*step	# now in seconds

		l = 1600000//R
		h, b = np.histogram(dt, bins=(np.arange(l+1)*R-0.5))
		l2 = 8000//R

		plt.figure()
		plt.plot((b[0:l2]+0.5)/R, h[0:l2]/h.sum(), "x", label=f"data for R={R}")

		
		p0 = [0.002, 400, 0.001, 2000]

		x = (b[int(T*DT)+1:l]+0.5)/R
		y = h[int(T*DT)+1:l]/h.sum()

		try: 
			params, _ = curve_fit(model, x, y, p0=p0, bounds=([0,0,0,0], [0.9,np.inf,0.2,np.inf]))
		except Exception as e:
			print(f"Fit failed: {e}")
			continue
		x2 = (b[int(T*DT)+1:l2]+0.5)/R
		plt.plot(x2, model(x2, *params), 'r', label=f"fit")

		a1, tau1, a2, tau2 = params

		print("found param:")
		print(f"a1 = ", a1)
		print(f"a2 = ", a2)
		print(f"tau1 = ", tau1)
		print(f"tau2 = ", tau2)

		plt.show()
		exit()

		pap, pdc1 = pap_sum_inf(T*DT, A1, u1, A2, u2, c)

		print("-----")
		print("proportion AP total", pap)
		print("pdc estimated", pdc1)
		print("-----")


		Pe = pdet1 / (F - f_rep*pdet1*DT*1e-6)
		print(f"We have (1- np.exp(-mu.eta) + Pdc)(1+pap) = ", Pe)
		print(f"Ie, with pap = {pap},  1- np.exp(-mu.eta) + Pdc = ", Pe/(1+pap))
		Pdmu = Pe / (1+pap) - Pdc
		print(f"From Pdc = {Pdc}")
		print(f"Pdmu = ", Pdmu)

		mu_deduced = -np.log(1 - Pdmu) / eta 
		print(f"Hence  mu_deduced = {mu_deduced} with eta = {eta}")
		print(f"or mu.eta = {-np.log(1-Pdmu)}")
		print('-----')


		Fcorr = (0.78 * (1 + pap) * Pdmu  + 0.26 * (1 + pap) * Pdc) / ((Pdmu + Pdc)*(1+pap))
		print(f"F corrigé = {Fcorr}")
		cdt0 = Fcorr / (1 + (Pdmu + Pdc)*(1+pap)*f_rep*DT*1e-6)
		print("cdt0 = ",cdt0)
		Nap = pap / (1+pap) * Nrecv
		print("Nap = ", Nap)
		Ndc = cdt0 * Pdc * Nsent   
		print("Ndc = ", Ndc)
		Nleg = Nrecv - Nap - Ndc
		print("Nleg = ", Nleg)


		# deduce tau1 tau2 a1 a2 
		pnc = 1-(1-np.exp(-MU*eta) + Pdc)
		tau1=1/(1/u1+R*np.log(pnc))
		tau2=1/(1/u2+R*np.log(pnc))

		sol = pnap(T*DT,tau1,tau2,A1,A2)
		print(f"sol = {sol}")
		a1 = sol[0][0]
		a2 = sol[0][1]
		if a1 > 1 or a2 > 1:
			a1 = sol[1][0]
			a2 = sol[1][1]
		if a1 > 1 or a2 > 1:
			print(f"no solution found {sol}")
			exit()

		print(F"--")
		print(f" a1 = ", a1)
		print(f" a2 = ", a2)

		print(f" tau1 = ", tau1)
		print(f" tau2 = ", tau2)

		#plt.plot(x2, pnextclick(x2,MU,eta,T*DT,a1,tau1,a2,tau2,Pdc), 'y', label=f"verif fit")
		
		p02 = [a1,tau1,a2,tau2,1e-5]

		try: 
			params, _ = curve_fit(model2, x, y, p0=p02, bounds=([0,0,0,0,0], [0.9,np.inf,0.12,np.inf,1]))
		except Exception as e:
			print(f"Fit failed: {e}")
			continue

		plt.plot(x2, model(x2, *params), label=f"optimized fit")

		a1, tau1, a2, tau2, c = params
		print(F"-- optimized param --")
		print(f" a1 = ", a1)
		print(f" a2 = ", a2)

		print(f" tau1 = ", tau1)
		print(f" tau2 = ", tau2)

		plt.plot(x2, pnextclick(x2,MU,eta,T*DT,a1,tau1,a2,tau2,Pdc), 'y', label=f"verif optimized fit")



		plt.legend()
		plt.title(f"{filename} and R={R}")


		fscan = np.linspace(0.26,0.5,3)
		Acolor={str(round(fscan[0],3)): 'orange', str(round(fscan[1],3)): 'violet', str(round(fscan[2],3)): 'red'}
		print(Acolor)

		#for d in [4, 6,10,20,100]:
		for d in [4,10]:
			fig = plt.figure()
			if d == 4:
				plt.plot([0.01,0.1,1.0],[15.4/100, 9.3/100, 4.2/100], "--", color='b', label=f"qber measured, d={d}us")
			else:
				plt.plot([0.01,0.1,1.0],[8.9/100, 6.0/100, 2.8/100], "--", color='b', label=f"qber measured, d={d}us")
			print("")
			print("DT=", d)
			for f in [fscan[0]]:
				print(f"f={f}")
				c=Acolor[str(round(f,3))]
				print(f"c={c}")
				for mu in [0.01,0.1,1.0]:
					print("mu=", mu)
					res_qber = compute_qber(d, mu, eta, a1, tau1, a2, tau2, Pdc, 0.02, f)
					print("res_qber = ", res_qber)
					plt.plot(mu,res_qber,'x', color=c, label=f"f={f}")
			plt.legend()
			plt.xlabel("mu")
			plt.title(f"check qber for {filename} and R={R}")



		plt.show()