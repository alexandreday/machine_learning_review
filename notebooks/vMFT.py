import numpy as np
import scipy.sparse as sp

n_rand=np.random.randint(10000000)
np.random.seed(np.random.randint(n_rand)) # fix random seed generator for reproducibility

print(n_rand)

##### Ising model parameters
Lx=40 # linear system size
Ly=40 # linear system size
Ns=Lx*Ly # number of sites
beta=2.0 # inverse temperatur

##### model
def couplings_2D(Lx,Ly):
	"""
	This function calculates the nn interaction strength J_{ij} and the magnetic field h_i 
	on a square lattice with periodic boundary conditions.

	"""
	J=np.zeros((Lx,Lx,Ly,Ly),)
	h=np.zeros((Lx,Ly),)
	for i in range(Lx):
		for j in range(Ly):
			h[i,j]=np.cos(i*2*np.pi/Lx)*np.cos(j*2*np.pi/Lx)
			for kl in [[0,-1],[1,0],[0,1],[-1,0]]:
					J[i,j,(i+kl[0])%Lx,(j+kl[1])%Ly]+=0.5
	
	J=sp.csc_matrix(J.reshape(Ns,Ns))
	h=h.reshape(Ns,)
	return J,h

# define couplings
J,h=couplings_2D(Lx,Ly)

##### solve self-consistency equation for b_j
# inittialize b
b=np.zeros((Ns,)) 
b[0]=0.0

# set epsilon to control iteration
eps=1.0 
i=0
while eps > 1E-9:
    b_new = beta*J.dot(np.tanh(b)) + h
    
    eps=np.max(np.abs(b-b_new))
    b=b_new.copy()
    i+=1

    if i > 1E4:
    	print('Convergence not reached within 1E4 iterations. Breaking loop.')
    	break

import matplotlib.pyplot as plt

#plt.plot(range(L),b)
cmap_args=dict(vmin=np.min(b), vmax=np.max(b), cmap='seismic')
plt.imshow(b.reshape(Lx,Ly),**cmap_args)
plt.colorbar() 
plt.show()