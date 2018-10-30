from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy import integrate
from scipy.integrate import quad,dblquad,nquad
import math


from scipy import *
from numpy import *
import numpy as np

from matplotlib import pyplot as plt

def main_func():
    interval = 0.05
    max= np.arange(0,2*np.pi+interval,interval)
    for idx in max:
        theta= np.arange(0,idx+0.1,0.1)
        xli01 = 16 * (sin(theta))**3
        yli01 = 13 * cos(theta) - 5 * cos(2 * theta) - 2 * cos(3 * theta) - cos(4 * theta)
        zrange=np.arange(0,5,5/len(theta))
        zli01= np.outer(zrange,exp(theta*0))

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111,projection='3d')
        ax.plot_surface(xli01,yli01,zli01,rstride=1, cstride=1, cmap=plt.cm.coolwarm)

        ax.set_xlim((-10, 10))
        ax.set_ylim((-10, 10))
        ax.set_zlim((0, 5))
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        name = 'test'+str(int(idx/interval))
        plt.savefig(name)
    return

