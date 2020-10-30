from IPython.display import clear_output
from matplotlib import pyplot as plt
import numpy as np
import collections
import time
%matplotlib inline


def live_plot(x):
    clear_output(wait=True)
    plt.plot(x,np.sin(x*0.5))
    plt.show()

if __name__ == '__main__':
    x = np.array([])
    for i in range(100):
        x = np.hstack([x,i])
        live_plot(x)
        time.sleep(0.01)
