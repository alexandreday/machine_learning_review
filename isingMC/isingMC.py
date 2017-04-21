'''
Author : Alexandre Day
Date : 2-feb-2017
Purpose : 
    Draws samples from 2D Ising model with periodic boundary conditions
    and uniform interactions J and longitudinal field hz
'''

import numpy as np
import sys

## average update time (delta_E function) is 2.4e-6 seconds <-

def main():
    
    Lx = 10
    Ly = 10
    L = Lx
    n_sample=40000
    n_spin = Lx*Ly
    hz=0.
    J=-1.0
    T = float(sys.argv[1]) # running from command line

    ising_model = Lattice2D(Lx=Lx, Ly=Ly, J=J, hz=hz)
    data=[]

    samples=MC(ising_model,beta=1./T, n_sample=n_sample)
    samples=np.array(samples).reshape(-1,Lx*Ly)
    samples[samples == -1] = 0 # replace -1 by zeros
    np.savetxt("mag_vs_T_L%i_T=%.2f.txt"%(L,T), samples,fmt="%i",delimiter=',')

    exit()

    for t in np.linspace(4.0,0.1,50):

        samples = MC(ising_model, beta=1./t, n_sample=n_sample) # Collecting samples at a given temperature
        magnetization = np.abs(np.sum(samples, axis=(1,2))) # Computing magnetization
        average_mag=np.mean(magnetization)/n_spin
        std_mag=np.std(magnetization/n_spin)/np.sqrt(n_sample) # Error on estimate
        data.append([t,average_mag,std_mag])
        print("Average magnetization at temperature T=%.3f :\t %.3f +- %.4f"%(t,average_mag, std_mag))
    
    print("Saving data")
    data=np.array(data)
    np.savetxt("mag_vs_T_L%i.txt"%L, data, delimiter='\t')

def MC(model, beta=0.3, n_sample=1000, reset=False):
    """
        Metropolis algorithm for drawing n_sample at inverse temperature beta.
        Note that beta_c = 0.44.
    """
    Lx,Ly=model.shape
    n_spin=Lx*Ly
    sample_list=[]

    n_equilibrate=20

    if reset:
        n_equilibrate=1000
        model.reset()

    for _ in range(n_spin * n_equilibrate): # First equilibrating system -> should be done with autocorrelation time ...
        spin_idx=np.unravel_index(np.random.randint(n_spin),dims=(Lx,Ly))
        dE=model.delta_E(spin_idx)
        if dE < 1e-10 :
            model.spin_state[spin_idx]*=-1.
        elif np.random.rand() < np.exp(-beta*dE) :
            model.spin_state[spin_idx]*=-1.

    for _ in range(n_sample):
        for _ in range(n_spin*5):
            spin_idx=np.unravel_index(np.random.randint(n_spin),dims=(Lx,Ly))
            dE=model.delta_E(spin_idx)
            if dE < 1e-10 :
                model.spin_state[spin_idx]*=-1.
            elif np.random.rand() < np.exp(-beta*dE) :
                model.spin_state[spin_idx]*=-1.
        sample_list.append(np.copy(model.spin_state))
    
    return np.array(sample_list)



class Lattice2D:
    """
    Class that encodes the spin state and the methods for computing
    energy given model parameters.
    """

    def __init__(self, Lx=2, Ly=2, J=-1., hz=0.):

        self.Lx, self.Ly = Lx, Ly
        self.shape =(Lx, Ly)
        self.spin_state= 2. * (np.round( np.random.rand(Lx, Ly) ) - 0.5) # Initiliaze lattice with spin states -1. or +1
        self.J=J
        self.hz=hz
        self.nn = np.empty(shape=(Lx, Ly), dtype=list)

        for i in range(Lx):
            for j in range(Ly):
                self.nn[i,j]=[[(i+1) % Lx, (i-1) % Lx, i, i], [j, j, (j+1) % Ly, (j-1) % Ly]] # Storing nearest-neighbor indices -> Top, Bottom, Left, Right

        grid = np.meshgrid(range(Lx),range(Ly))
        self.grid_pos=np.vstack(map(np.ravel, grid)).T # An array of shape (nspin,2) corresponding the the coordinates of the spins
    
    def __getitem__(self,key): 
        return self.spin_state[key]

    def __str__(self): # print spin configuration
        return str(self.spin_state)

    def energy(self):

        '''Computes total energy of lattice E = J*s_i*s_j+hz*s_i (summed over indices)'''
        
        EJ, Eh = (0., 0.)
        for i,j in self.grid_pos: # This could be vectorized, but for clarity is left as a for loop ! Marin: YES, look up np.einsum!
            sij=self.spin_state[i,j]
            snn=self.spin_state[self.nn[i,j]] # nn spins
            EJ += sij*np.sum(snn)
            Eh += sij

        return 0.5 * EJ * self.J + Eh * self.hz

    def delta_E(self,ij): # this returns the difference in energy upon flipping spin at position (i,j)
        return -2. * self.J * self.spin_state[ij]*np.sum(self.spin_state[self.nn[ij]])

    def magnetization(self): # Computes magnetization of the current model spin state
        return np.sum(self.spin_state)
    
    def reset(self): # Resets the spin state randomly (high-temperature state)
        self.spin_state= 2. * (np.round( np.random.rand(self.Lx, self.Ly) ) - 0.5) # Initiliaze lattice with spin states -1. or +1


if __name__=="__main__":
    main()