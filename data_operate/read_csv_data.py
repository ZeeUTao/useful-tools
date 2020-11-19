import csv
import numpy as np

file_name = r'X:\xxx\xxx.csv'

# read via csv
ifile = open(file_name, "r")
reader = csv.reader(ifile)
data = np.asarray(list(reader),dtype='float')

# read via numpy
data = np.loadtxt(file_name,delimiter=',')
