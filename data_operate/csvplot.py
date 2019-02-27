import csv
from matplotlib import pyplot as plt
import numpy as np


csv_reader = csv.reader(open("00272 - %vq2%g%c spectroscopy.csv"))

csv_row = [row for row in csv_reader]

csv_row = np.array(csv_row,dtype=float)

data = [csv_row[:,0],csv_row[:,1],csv_row[:,2]]

# plt.scatter(data[0],data[1],c=data[2])
 
data_shape = (151,211)

for i in range(len(data)):
    data[i] = data[i].reshape(data_shape)
    
plt.pcolor(data[0],data[1],data[2]/np.max(data[2]),cmap =plt.get_cmap('Greens'))
plt.xlabel('Z-pulse amplitude')
plt.ylabel('XY-pulse frequency')
plt.tight_layout()
plt.colorbar()