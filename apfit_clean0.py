import numpy as np 
import numpy as np, matplotlib.pylab as plt
from scipy.optimize import curve_fit
import sys

Lqber = np.array([50, 15.4, 9.3, 4.2, 50, 8.9, 6.0, 2.8])/100
Lmu = [0.0, 0.01, 0.1, 1.0, 0.01, 0.1, 1.0]
Ldt = [4,4,4,4,10,10,10,10]
Rlist = [1,2,4,5,8,10,16,20,40,80]
#Best value : 
Rlist=[16]

print(Lqber)

print("With paramters :")
f_rep = 80e6
minimum_attenuation = 20
dkm = 0
eta = 10**(-(minimum_attenuation + dkm *0.23)/10)
F = 0.7 #coeff gate size
Pdc = 1.33e-6

print(f"eta = {eta}")
print(f"F = {F}")
print(f"Pdc = {Pdc}")

def model2(x, a1, tau1, a2, tau2, a3, tau3, noiseap):
	#Pnc = np.exp(-MU*eta) - Pdc_ref
	Pc0 = 1-np.exp(-MU*eta) + Pdc
	cdt = F / (1 + Pc0*f_rep*DT*1e-6)
	Pnc = 1- F*Pc0
	return Pnc**(R*(x-T*DT))*(Pnc**R*(a1 * np.exp(-x * (1 / tau1)) + a2 * np.exp(-x * (1 / tau2)) + a3 * np.exp(-x * (1 / tau3))) + cdt * Pdc  + cdt * (1-np.exp(-MU*eta)) + noiseap)
	



def pap_sum_inf(dt, A1, u1, A2, u2, A3, u3, c):
	#F = 1 / (1-np.exp(-mu*eta)) # "legitimate window" in UT (average time between 2 legitimate pulses)
	pap1 = A1 * np.exp(-(dt+1)/u1) / (1 - np.exp(-1/u1))
	pap2 = A2 * np.exp(-(dt+1)/u2) / (1 - np.exp(-1/u2))
	pap3 = A3 * np.exp(-(dt+1)/u3) / (1 - np.exp(-1/u3)) 
	pdc = c 
	return (pap1 + pap2 + pap3), pdc

def model(x, a1, tau1, a2, tau2, a3, tau3, c):
	return a1 * np.exp(-x / tau1) + a2 * np.exp(-x / tau2) + a3 * np.exp(-x / tau3) + c

def test_f(x,mu,eta,a1,tau1,a2,tau2,a3,tau3,c) :
	#pnc = np.exp(-mu*eta*80*x)
	#pnc = (np.exp(-mu*eta) - c )**(R*x)
	pnc = (1-F*(1-np.exp(-mu*eta) + c))**(R*x)
	pap_n = a1*np.exp(-x/tau1)+a2*np.exp(-x/tau2)+a3*np.exp(-x/tau3)
	pdc = c + 1-np.exp(-mu*eta)
	return pnc*(pap_n+pdc) 


def qber_DT(dt,mu,eta,A1, tau1, A2, tau2, A3, tau3, pdc, e0):
	pap = Pap_DT(dt, mu,eta,A1, tau1, A2, tau2, A3, tau3)
	print("pap = ", pap) 
	#Pdt = (1 - np.exp(-mu*eta) + pdc) * (1+pap)
	Pdt = F*(1 - np.exp(-mu*eta) + pdc) * (1+F*pap)
	return (e0 + pap*0.5), (F*(1-np.exp(-mu*eta))*(e0 + F*pap*0.5) + F*pdc*0.5*(1 + F*pap)) / Pdt
	

def Pap_DT(dt, mu, eta, A1, tau1, A2, tau2, A3, tau3):
	Pc0 = 1-np.exp(-mu*eta) + Pdc
	#cdt = F / (1 + Pc0*f_rep*dt*1e-6)
	#cdt = 0.6*F
	#print(f"cdt {cdt} Pc0 {Pc0}")
	#Pc = cdt * Pc0
	#Pnc = np.exp(-mu*eta) - Pdc
	Pnc = 1 - F*Pc0
	#print(f"Pnc {Pnc}")
	dt1=T*dt # Deadtime for divided Unit of time
	#print(f"deadtime per unit considered {dt1}")
   
	R1=R*(1-dt1)
	u1 = 1/ (1/tau1 - R*np.log(Pnc))
	u2 = 1/ (1/tau2 - R*np.log(Pnc))
	u3 = 1/ (1/tau3 - R*np.log(Pnc))

	# pap1 = Pnc**(R*(1-DT))*A1 * np.exp(-(DT+1)/u1) / (1 - np.exp(-1/u1)) 
	# pap2 = Pnc**(R*(1-DT))*A2 * np.exp(-(DT+1)/u2) / (1 - np.exp(-1/u2)) 
	# pap3 = Pnc**(R*(1-DT))*A3 * np.exp(-(DT+1)/u3) / (1 - np.exp(-1/u3)) 
	pap1 = Pnc**R1*A1 * np.exp(-(dt1+1)/u1) / (1 - np.exp(-1/u1)) 
	pap2 = Pnc**R1*A2 * np.exp(-(dt1+1)/u2) / (1 - np.exp(-1/u2)) 
	pap3 = Pnc**R1*A3 * np.exp(-(dt1+1)/u3) / (1 - np.exp(-1/u3)) 

	return (pap1 + pap2 + pap3)

#filename = ["dt10_mu0.01.txt"]
#labelname = [" dt"] 

#filename = sys.argv[1]

#"dt4_mu0.txt"/"dt10_mu1.txt" : can t fit
#"dt4_mu1.0.txt","dt4_mu0.1.txt", "dt10_mu0.txt",   "dt10_mu0.1.txt"
#"dt10_mu0.01.txt"

#best value only
for filename in ["dt4_mu0.01.txt"]:
	print("")
	print(f"Filename = {filename}")

	if "dt10_" in filename:
		DT = 10 # UT
	elif "dt4_" in filename:
		DT = 4 # UT
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

	if MU == 0.0:
		print(f"Estimate DC for mu=0:")
		print("DC <= ", pdet1)
		Pdc_ref = pdet1 /2
		print(f"Pdc_ref = {Pdc_ref} (Attention, F and DT is incuded, otherwise Pdc_ref_woF = {Pdc_ref/F})")
		print(f"DC with Pdc_ref {Pdc_ref} = ", Pdc_ref*Nsent)



	
	


	for R in Rlist:
		T = 80/R # division of UT, ie a unit is 1/T microsecond
		print(f"R = {R}")


		#dt = dt*step	# now in seconds

		l = 1600000//R
		h, b = np.histogram(dt, bins=(np.arange(l+1)*R-0.5))
		l2 = 8000//R

		plt.figure()
		plt.plot(b[0:l2]/R+0.5, h[0:l2]/h.sum(), "x", label=f"data for R={R}")




		p0 = [0.9, 1, 0.1, 10, 0.01, 100, 1.1e-3]


		x = b[int(T*DT)+1:l]/R
		y = h[int(T*DT)+1:l]/h.sum()

		try: 
			params, _ = curve_fit(model, x, y, p0=p0, bounds=([0,0,0,0,0,0,0], [0.9,np.inf,0.1,np.inf,0.01,np.inf,1]))
		except Exception as e:
			print(f"Fit failed: {e}")
			continue
		x2 = b[int(T*DT)+1:l2]/R
		plt.plot(x2+0.5, model(x2, *params), 'r', label=f"fit")

		a1, u1, a2, u2, a3, u3, c = params

		print("found param:")
		print(f"a1 = ", a1)
		print(f"a2 = ", a2)
		print(f"a3 = ", a3)

		print(f"tau1 = ", u1)
		print(f"tau2 = ", u2)
		print(f"tau3 = ", u3)

		pap, pdc1 = pap_sum_inf(T*DT, a1, u1, a2, u2, a3, u3, c)

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
		print(f"-----\nHence  mu_deduced = {mu_deduced} with eta = {eta}")
		print(f"or mu.eta = {-np.log(1-Pdmu)}")
		print('-----')


		cdt0 = F / (1 + (Pdmu + Pdc)*(1+pap)*f_rep*DT*1e-6)
		print("cdt0 = ",cdt0)
		Nap = pap / (1+pap) * Nrecv
		print("Nap = ", Nap)
		Ndc = cdt0 * Pdc * Nsent   
		print("Ndc = ", Ndc)
		Nleg = Nrecv - Nap - Ndc
		print("Nleg = ", Nleg)


		#Pnc = np.exp(-MU*eta) - Pdc
		Pnc = 1-F*(1-np.exp(-MU*eta) + Pdc)
		tau1=1/(1/u1+R*np.log(Pnc))
		tau2=1/(1/u2+R*np.log(Pnc))
		tau3=1/(1/u3+R*np.log(Pnc))

		print(f"With R={R}")
		R1=R*(1-T*DT)
		a11=a1/Pnc**R1
		a22=a2/Pnc**R1
		a33=a3/Pnc**R1

		print(f"a11 = ", a11)
		print(f"a22 = ", a22)
		print(f"a33 = ", a33)

		print(f"tau1 = ", tau1)
		print(f"tau2 = ", tau2)
		print(f"tau3 = ", tau3)


		plt.plot(x2+0.5, test_f(x2,MU,eta,a1,tau1,a2,tau2,a3,tau3,1e-6), 'y', label=f"verif fit")
		#plt.plot(x+0.5, test_f(x,MU/5,eta,a1,tau1,a2,tau2,a3,tau3,1e-6), label="verif 2")
		#plt.plot(x+0.5, test_f(x,MU/10,eta,a1,tau1,a2,tau2,a3,tau3,1e-7), label="verif 3")

		# opptimize the fit
		p02=[a11, tau1, a22, tau2, a33, tau3, 1e-5]
		try: 
			params2, _ = curve_fit(model2, x, y, p0=p02, bounds=([0,0,0,0,0,0,0], [1,np.inf,1,np.inf,1,np.inf,1]))
		except Exception as e:
			print(f"Fit failed: {e}")
			continue

		a111, tau111, a222, tau222, a333, tau333, noiseap = params

		print(f"a111 = ", a111)
		print(f"a222 = ", a222)
		print(f"a333 = ", a333)

		print(f"tau111 = ", tau111)
		print(f"tau222 = ", tau222)
		print(f"tau333 = ", tau333)


		plt.plot(x2+0.5, model2(x2, *params2), 'violet', label=f"optimize fit")
		plt.legend()
		plt.title(f"{filename} and R={R}")



		#for d in [4, 6,10,20,100]:
		for d in [4,10]:
			fig = plt.figure()
			if d == 4:
				plt.plot([0.01,0.1,1.0],[15.4/100, 9.3/100, 4.2/100], label=f"qber measured, d={d}us")
			else:
				plt.plot([0.01,0.1,1.0],[8.9/100, 6.0/100, 2.8/100], label=f"qber measured, d={d}us")
			print("")
			print("DT=", d)
			for mu in [0.01,0.1,1.0]:
				print("mu=", mu)
				_ , res_qber = qber_DT(d,mu,eta,a111, tau111, a222, tau222, a333, tau333, Pdc, e0=0.02)
				print("res_qber = ", res_qber)
				plt.plot(mu,res_qber,'x')
			plt.legend()
			plt.title(f"check qber for {filename} and R={R}")



		plt.show()