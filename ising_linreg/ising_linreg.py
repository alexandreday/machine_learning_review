import numpy as np
import matplotlib.pyplot as plt

import sys

np.random.seed(0)

def main():
    # define system size
    Lx, Ly = 40, 40
    Ns = Lx*Ly
    n_spin = Lx*Ly
    # define Ising model aprams
    h=0.0 # magnetic field
    # interactions strength
    J=np.zeros((Lx,Ly,Lx,Ly),)
    for i in range(Lx):
        for j in range(Ly):
            for k in [-1,1]:
                for l in [-1,1]:
                    J[i,j,(i+k)%Lx,(j+l)%Ly]=-1.0

    # command line input
    T = float(sys.argv[1]) # temperature

    # load data
    samples=np.loadtxt("mag_vs_T_L%i_T=%.2f.txt" %(Lx,T),delimiter=",",dtype=np.int)
    # pre-process data
    samples[np.where(samples==0)]=-1 # replace 0 by -1
    samples=samples.reshape(10000,Lx,Ly) # expand states in 2D square lattice format
    print('finished loading and pre-processing data...')

    # build Ising model
    ising_model = Ising2D(Lx=Lx,Ly=Ly,J=J,h=h,states=samples)
    # calculate Ising energies of spin states
    energies=ising_model.energy()

    print('Emin and Emax:', min(energies)/Ns, max(energies)/Ns)
    print('samples E-density:', (min(energies)+T)/Ns)

    #exit()

    plt.hist(energies/Ns,bins=50)
    plt.show()

    exit()


class Ising2D:
    """
    Class that encodes the spin state and the methods for computing
    energy given model parameters.

    """

    def __init__(self, Lx=2, Ly=2, J=-1.0, h=0.0, states=None):

        ### check input variables
        if not isinstance(J,np.ndarray):
            J=np.asarray(J)

        if not isinstance(h,np.ndarray):
            h=np.asarray(h)

        if states is None:
            self.spin_state= 2. * (np.round( np.random.rand(Lx, Ly) ) - 0.5) # Initiliaze lattice with spin states -1. or +1
        else:
            self.spin_state=states

        ### define system size
        self.Lx, self.Ly = Lx, Ly
        self.shape=(Lx, Ly)
        self.Ns=Lx*Ly

        ### define model params
        # 2D interaction strength: 4d array (x,y,x,y)
        if J.shape==(Lx,Ly,Lx,Ly):
            self.J=J
        else:
            self.J=J.reshape(1,1,1,1) 
        # 2D magnetic field: 2d array (x,y)
        if h.shape==(Lx,Ly):
            self.h=(Lx,Ly)
        else:
            self.h=h.reshape(1,1) 
        
        ### define lattice grid
        grid = np.meshgrid(range(Lx),range(Ly))
        self.grid_pos=np.vstack(map(np.ravel, grid)).T # An array of shape (nspin,2) corresponding the the coordinates of the spins
            

    def __getitem__(self,key): 
        return self.spin_state[key]

    def __str__(self): # print spin configuration
        return str(self.spin_state)


    def _pot_energy(self):
        '''Computes total potential energy of state E = \sum_{i} h_i*s_i'''
        return np.einsum('ij,...ij->...',self.h,self.spin_state)

    def _int_energy(self):
        '''Computes total interaction energy of state E = \sum_{i,j} J_{ij}*s_i*s_j'''
        return 0.5*np.einsum('...ij,ijkl,...kl->...',self.spin_state,self.J,self.spin_state)

    def energy(self,normed=False):
        '''Computes total energy of lattice E = J*s_i*s_j+h*s_i (summed over indices)'''
        if normed:
            E=(self._int_energy() + self._pot_energy() )/self.Ns
        else:
            E=self._int_energy() + self._pot_energy()
        
        return E

    def magnetization(self): # Computes magnetization of the current model spin state
        if self.spin_state.shape==(Lx,Ly):
            M=np.sum(self.spin_state)
        else:
            M=np.sum(self.spin_state,axis=(1,2))
        
        return M
    

if __name__=="__main__":
    main()
