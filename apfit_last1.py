import numpy as np 
import numpy as np, matplotlib.pylab as plt
from scipy.optimize import curve_fit
from sympy import symbols, solve
import sys


## this version only approximate pnap : proba of no apfter pulse from dt to n with pnap = np.exp(-L1 -L2)
## and compute the "infinite" sum with a for loop


Lqber = np.array([50, 15.4, 9.3, 4.2, 50, 8.9, 6.0, 2.8])/100
Lmu = [0.0, 0.01, 0.1, 1.0, 0.01, 0.1, 1.0]
Ldt = [4,4,4,4,10,10,10,10]
Rlist = [1,2,4,5,8,10,16,20,40,80]
Rlist = [16,20,40,80]
#Best value : 
Rlist=[1, 80]

print(Lqber)

print("With paramters :")
f_rep = 80e6
minimum_attenuation = 20
dkm = 0
eta = 10**(-(minimum_attenuation + dkm *0.23)/10)
eta = 0.01
F = 0.78 #coeff gate size
Pdc = 5e-6

print(f"eta = {eta}")
print(f"F = {F}")
print(f"Pdc = {Pdc}")


def pn(n, nd, a1, tau1, a2, tau2, mueta):
	nd = nd//R
	pc = 1-np.exp(-mueta) + Pdc
	pnc = 1- pc
	pap = a1*np.exp(-n * (1 / tau1)) + a2 *np.exp(-n * (1 / tau2))
	B1 = a1/(1-np.exp(-1/tau1))
	B2 = a2/(1-np.exp(-1/tau2))
	L1 = B1*np.exp(-(nd + 1)/tau1)*(1-np.exp(-(n - nd - 1) / tau1))
	L2 = B2*np.exp(-(nd + 1)/tau2)*(1-np.exp(-(n - nd - 1) / tau2))
	pnap = np.exp(-L1 -L2)
	#pnap = 1
	return pnc**(R*(n-nd-1)) * pnap * (pap + (1-pnc**R))



def pnpap(n, nd, a1, tau1, a2, tau2, mueta):
	nd = nd//R
	pc = 1-np.exp(-mueta) + Pdc
	pnc = 1- pc
	pap = a1* np.exp(-n * (1 / tau1)) + a2 *np.exp(-n * (1 / tau2))
	B1 = a1/(1-np.exp(-1/tau1))
	B2 = a2/(1-np.exp(-1/tau2))
	L1 = B1*np.exp(-(nd + 1)/tau1)*(1-np.exp(-(n - nd - 1) / tau1))
	L2 = B2*np.exp(-(nd + 1)/tau2)*(1-np.exp(-(n - nd - 1) / tau2))
	pnap = np.exp(-L1 -L2)
	#pnap = 1
	return pnc**(R*(n-nd-1)) * pnap * (pap)


def pnlegdc(n, nd, a1, tau1, a2, tau2, mueta):
	nd = nd//R
	pc = 1-np.exp(-mueta) + Pdc
	pnc = 1- pc
	pap = a1* np.exp(-n * (1 / tau1)) + a2 *np.exp(-n * (1 / tau2))
	B1 = a1/(1-np.exp(-1/tau1))
	B2 = a2/(1-np.exp(-1/tau2))
	L1 = B1*np.exp(-(nd + 1)/tau1)*(1-np.exp(-(n - nd - 1) / tau1))
	L2 = B2*np.exp(-(nd + 1)/tau2)*(1-np.exp(-(n - nd - 1) / tau2))
	pnap = np.exp(-L1 -L2)
	return pnc**(R*(n-nd-1)) * pnap * (1-pnc**R)


def model(n, a1, tau1, a2, delta2, mueta):
	nd = DT//R
	pc = 1-np.exp(-mueta) + Pdc
	pnc = 1- pc
	tau2 = tau1 + delta2
	pap = a1* np.exp(-n * (1 / tau1)) + a2 * np.exp(-n * (1 / tau2))
	B1 = a1/(1-np.exp(-1/tau1))
	B2 = a2/(1-np.exp(-1/tau2))
	L1 = B1*np.exp(-(nd + 1)/tau1)*(1-np.exp(-(n - nd - 1) / tau1))
	L2 = B2*np.exp(-(nd + 1)/tau2)*(1-np.exp(-(n - nd - 1) / tau2))
	#pnap = 1 - L1 - L2 
	pnap = np.exp(-L1 -L2)
	#pnap = 1
	return pnc**(R*(n-nd-1)) * pnap * (pap + (1-pnc**R))

def compute_pap(nd, mueta, a1, tau1, a2, tau2, dtmax):
	sum1 = 0
	for i in np.arange((nd)//R + 1, int(dtmax//R)):
		sum1 += pnpap(i, nd, a1, tau1, a2, tau2, mueta)
	return sum1

def compute_qber(dt, mu, eta, a1, tau1, a2, tau2, pdc, e0, foF, dtmax):
	pap = compute_pap(dt, mu*eta, a1, tau1, a2, tau2, dtmax)
	print(f"pap = {pap}")
	pq = 1 - np.exp(-mu*eta)
	pap_pulse = pap * (pq + pdc) / (1 - pap)
	return (foF * pap_pulse * 0.5 + pq * e0 + foF * pdc * 0.5) / (foF*pap_pulse + pq + foF * pdc) 


def model0(x, a1, tau1, a2, tau2, c):
	return a1 * np.exp(-x / tau1) + a2 * np.exp(-x / tau2) + c


def summ(alpha, n): 
	return np.exp(-alpha*(n+1))/(1 - np.exp(-alpha))

def compute_pap0(nd, mueta, a1, tau1, a2, tau2):
	nd = nd // R
	pc = 1-np.exp(-mueta) + Pdc
	pnc = 1- pc
	B1 = a1/(1-np.exp(-1/tau1))
	B2 = a2/(1-np.exp(-1/tau2))
	T1 = 1 - B1*np.exp(-(nd + 1)/tau1) - B2*np.exp(-(nd + 1)/tau2)
	S1 = summ(1/tau1 - R*np.log(pnc), nd)
	S2 = summ(1/tau2 - R*np.log(pnc), nd)
	S3 = summ(2/tau1 - R*np.log(pnc), nd)
	S4 = summ(1/tau1 + 1/tau2 - R*np.log(pnc), nd)
	S5 = summ(1/tau1 + 1/tau2 - R*np.log(pnc), nd)
	S6 = summ(2/tau2 - R*np.log(pnc), nd)
	
	return pnc**(R*(-nd-1)) * (
		  T1 * ( a1 * S1 + a2 * S2 ) 
		+ B1 * ( a1 * S3 + a2 * S4 )
		+ B2 * ( a1 * S5 + a2 * S6 )
		)
	
def compute_pdet10(nd, mueta, a1, tau1, a2, tau2):
	nd = nd//R
	pc = 1-np.exp(-mueta) + Pdc
	pnc = 1- pc
	B1 = a1/(1-np.exp(-1/tau1))
	B2 = a2/(1-np.exp(-1/tau2))
	T1 = 1 - B1*np.exp(-(nd + 1)/tau1) - B2*np.exp(-(nd + 1)/tau2)
	S1 = summ(- R*np.log(pnc), nd)
	S2 = summ(1/tau1 - R*np.log(pnc), nd)
	S3 = summ(1/tau2 - R*np.log(pnc), nd)
	
	return pnc**(R*(-nd-1)) * (
		  T1 * ( (1-pnc**R) * S1 ) 
		+ B1 * ( (1-pnc**R) * S2 )
		+ B2 * ( (1-pnc**R) * S3 )
		)

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
		DT = 80*10 # UT
	elif "dt4_" in filename:
		DT = 80*4 # UT
	elif filename == "wide_gates.txt":
		DT = 337
		DEADTIME = 337
		#p0 = [0.9, 1, 0.1, 10]
	else:
		print("no known DT")
		sys.exit()


	# if "mu0.t" in filename:
	# 	MU = 0 
	# elif "mu0.01" in filename:
	# 	MU = 0.01
	# elif "mu0.1" in filename:
	# 	MU = 0.1
	# elif "mu1." in filename:
	# 	MU = 1.0
	# elif filename == "wide_gates.txt":
	# 	MU = 0.005
	# else:
	# 	print("no known MU")
	# 	sys.exit()


	print("DT = ", DT)
	#print("MU = ", MU)


	data = np.loadtxt(filename)
	data = data[data!=0]
	dt = data[1:] - data[:-1]

	Nsent = data[-1]
	print(f"Nsent : ", Nsent)
	Nrecv = len(data)
	print(f"total count", Nrecv)
	pdet1= Nrecv/Nsent
	print(f" pdet1 = {pdet1}")
	# if MU != 0: 
	# 	eta_deduced = pdet1 / ((F - pdet1*f_rep*DT*1e-6)*MU)
	# 	print(f"If mu = 1 (ie negl DC and AP), eta_deduced = {eta_deduced}, must be >= eta_ref = {eta}")

	# #if MU == 0.0:
	# print(f"Estimate DC for mu=0:")
	# print("  DC <= ", pdet1)
	# Pdc_ref = pdet1 /2
	# print(f"  Pdc_ref = {Pdc_ref} (Attention, F and DT is incuded, otherwise Pdc_ref_woF = {Pdc_ref/F})")
	# print(f"  DC with Pdc_ref {Pdc_ref} = ", Pdc_ref*Nsent)



	
	


	for R in Rlist:
		# division of UT, ie a unit is 1/T microsecond
		print(f"R = {R}")


		#dt = dt*step	# now in seconds

		#l = 1600000//R
		l = int(dt.max()//R + 1)
		h, b = np.histogram(dt, bins=(np.arange(l+1)*R-0.5))
		l2 = 8000//R

		plt.figure()
		plt.plot((b[0:l2]+0.5)/R, h[0:l2]/h.sum(), "x", label=f"data for R={R}")

		# plt.legend()
		# plt.show()
		# exit()
		if filename == "wide_gates":
			if R == 1:		
				#p0 = [0.002, 400, 0.001, 2000, 0.01]
				p0 = [0.002, 400, 0.001, 1600, 0.01]
			elif R == 16 : 
				p0 = [0.025, 33, 0.001, 148,0.01]
			else : 
				p0 = [0.002*R, 400//R, 0.001, 2000//R, 0.01]
		else:
			p0 = [0.1, 1, 0.01, 9, 0.01] 


		x = (b[int(DT//R)+1:l]+0.5)/R
		y = h[int(DT//R)+1:l]/h.sum()

		try: 
			params, _ = curve_fit(model, x, y, p0=p0, bounds=([0,0,0,0,0], [0.9,np.inf,0.2,np.inf,1]))
		except Exception as e:
			print(f"Fit failed: {e}")
			continue
		x2 = (b[int(DT//R)+1:l2]+0.5)/R

		plt.plot(x2, model(x2, *params), 'r', label=f"fit")

		print(np.sum(model(x2,*params)))

		a1, tau1, a2, delta2, mueta = params
		tau2 = tau1 + delta2

		print("found param:")
		print(f"a1 = ", a1)
		print(f"a2 = ", a2)
		print(f"tau1 = ", tau1)
		print(f"tau2 = ", tau2)
		print(f"mueta = ", mueta)

		# print("optimize fit")
		# p0 = [a1,tau1, a2, tau2, mueta]

		# try: 
		# 	params, _ = curve_fit(model, x, y, p0=p0, bounds=([0,0,0,0,0], [0.9,np.inf,0.2,np.inf,1]))
		# except Exception as e:
		# 	print(f"Fit failed: {e}")
		# 	continue
		# x2 = (b[int(DT//R)+1:l2]+0.5)/R
		# plt.plot(x2, model(x2, *params), 'violet', label=f"optimize fit")

		# a1, tau1, a2, tau2, mueta = params

		# print("found param:")
		# print(f"a1 = ", a1)
		# print(f"a2 = ", a2)
		# print(f"tau1 = ", tau1)
		# print(f"tau2 = ", tau2)
		# print(f"mueta = ", mueta)



		plt.legend()
		plt.title(f"{filename} and R={R}")

		# pdet1_deduced = compute_pdet1(DT, mueta, a1, tau1, a2, tau2)
		# pap_deduced = compute_pap(DT, mueta, a1, tau1, a2, tau2)
		# print(f"pdet1 deduced = {pdet1_deduced}")
		# print(f"pap deduced = {pap_deduced}")	
		# print(f"total = {pdet1_deduced + pap_deduced}")

		print(f"verif")
		# sum1 = 0
		sum2 = 0
		sum1 = 0
		for i in np.arange(DEADTIME//R+1, int(dt.max()//R)):
			sum1 += pn(i, DEADTIME, a1, tau1, a2, tau2, mueta)
			sum2 += pnpap(i, DEADTIME, a1, tau1, a2, tau2, mueta)
		# 	sum3 += pnlegdc(i, DEADTIME, a1, tau1, a2, tau2, mueta)
		print(f"sum total = {sum1}")
		print(f"sum pap = {sum2}")
		# print(f"sum leg + dc  = {sum3}")
		# dtr = int(DT//R+1)
		# print(f"h[{dtr}:].sum()/h.sum() = {h[dtr:].sum()/h.sum()}")



		fscan = np.array([0.2, 0.26/0.78, 0.7])
		Acolor={str(round(fscan[0],3)): 'orange', str(round(fscan[1],3)): 'violet', str(round(fscan[2],3)): 'red'}
		print(Acolor)

		#for d in [4, 6,10,20,100]:
		for d in [4, 10]:
			fig = plt.figure()
			if round(d,1) == 4:
				plt.plot([0.01,0.1,1.0],[15.4/100, 9.3/100, 4.2/100], "--", color='b', label=f"qber measured, d={d}us")
			else:
				plt.plot([0.01,0.1,1.0],[8.9/100, 6.0/100, 2.8/100], "--", color='b', label=f"qber measured, d={d}us")
			print("")
			print("DT=", d)
			for foF in [fscan[1]]:
				print(f"foF={foF}")
				c=Acolor[str(round(foF,3))]
				print(f"c={c}")
				for mu in [0.01,0.1,1.0]:
					print("mu=", mu)
					#print(f"pdet1= {compute_pdet1(80*d, mu*eta, a1, tau1, a2, tau2)}")
					res_qber = compute_qber(80*d, mu, eta, a1, tau1, a2, tau2, Pdc, 0.02, foF, dt.max())
					print("res_qber = ", res_qber)
					plt.plot(mu,res_qber,'x', color=c, label=f"foF={foF}")
			plt.legend()
			plt.xlabel("mu")
			plt.title(f"check qber for {filename} and R={R} and d={d}us")



		plt.show()