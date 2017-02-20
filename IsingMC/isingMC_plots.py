
from matplotlib import pyplot as plt
import matplotlib as mpl
from ml_style import ml_style_2 as sty
mpl.rcParams.update(sty.style)
import numpy as np

for l in [8,12,16,20,24]:
    data=np.loadtxt("mag_vs_T_L%i.txt"%l, delimiter='\t')
    #print("Plotting results")
    plt.scatter(data[:,0],data[:,1],label="$L=%i$"%l)
plt.xlabel("$T$")
plt.legend(loc='best')
#plt.tight_layout()
plt.ylabel("Magnetization")

J=-1
hz=0.0
n_sample=500
plt.title("Magnetization as a function of temperature for the 2D Ising model \n ($J=%.2f, h_z=%.2f$, \# sample $= %i$)"%(J,hz,n_sample))
plt.show()