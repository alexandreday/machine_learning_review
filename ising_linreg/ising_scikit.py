import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

#from sklearn import cross_validation, datasets, linear_model

n_samples=1000

# system size
Lx=20
Ly=20
Ns=Lx*Ly
# temperature
T=4.0

states_str = "mag_vs_T_L%i_T=%.2f.txt" %(Lx,T)
	
states=np.loadtxt(states_str,delimiter=",",dtype=np.int)

S = states
SS = np.einsum('...i,...j->...ij', S, S)

X = SS.reshape((n_samples,Ns*Ns))


##### model 1: J nn, single parameter
J=np.zeros((Lx,Ly,Lx,Ly),)
for i in range(Lx):
	for j in range(Ly):
		for k in [-1,1]:
			for l in [-1,1]:
				J[i,j,(i+k)%Lx,(j+l)%Ly]-=1.0
J=J.reshape((Ns*Ns,))

X = X.dot(J)

print('M1: dim X', X.shape)
print('M1: dim W', (1,))

##### model 2: J nn, all different symmetric
X = SS.reshape((n_samples,Ns*Ns))
W = 0.5*(J+J.T).reshape((Ns*Ns,))
W_sp = sp.csr_matrix(W)

inds_nn=W_sp.nonzero()[1]
X_nn = X[:,inds_nn] 
W_nn = W_sp.data

print('M2: dim X', X_nn.shape)
print('M2: dim W', W_nn.shape)

##### model 3: J nn, all different no symmetry
#X = SS.reshape((n_samples,Ns*Ns))
W = J.reshape((Ns*Ns,))
W_sp = sp.csr_matrix(W)

inds_nn=W_sp.nonzero()[1]
X_nn = X[:,inds_nn] 
W_nn = W_sp.data

print('M3: dim X', X_nn.shape)
print('M3: dim W', W_nn.shape) 

##### model 4: J fully connected symmetric
#X = SS.reshape((n_samples,Ns*Ns))
W = 0.5*(J+J.T).reshape((Ns*Ns,))

print('M4: dim X', X.shape)
print('M4: dim W', W.shape) 

##### model 5: J fully connected no symmetry
#X = SS.reshape((n_samples,Ns*Ns))
W = J.reshape((Ns*Ns,))

print('M4: dim X', X.shape)
print('M4: dim W', W.shape) 


