# -*- coding: utf-8 -*-

import numpy as np

from numpy import sin,cos,sqrt,arctan, pi, exp 
import matplotlib.pyplot as plt
import pylab as pl
from qutip import *
from scipy import integrate
from scipy.integrate import quad,dblquad,nquad
import math
from scipy import optimize

# import time to name the file
import datetime
nowTime=datetime.datetime.now().strftime('%Y%m%d %H%M%S') #现在时间

V0 = 5.0   # Volt
F0 = 209   # Hertz
#Import data
data = np.genfromtxt('FS2.txt',delimiter='//',skip_header=2)   #skip header
# C1-C8
data_x = data[:,0]
data_y = data[:,1]


def area_raw(xlist,ylist):
    area = 0
    for idx,idx_x in enumerate(xlist):
        if idx_x>190 and idx_x<218:
            area = area + (xlist[idx+1] - xlist[idx])*(0.7746/44 * (np.exp(ylist[idx])- np.exp(ylist[185])))
    return area

def fitting_fig(draw):
    interval = 0.1
    new_xlist =data_x
    def func(x,A,sigma,mu,B):
        return A*(1/np.sqrt(2*np.pi)/sigma)*np.exp(-  (x-mu)**2   /2/sigma**2    ) + B

    fita, fitb = optimize.curve_fit(func, new_xlist, data_y, [10e-3,-13.222923164,209,-59.7381196783])


    new_ylist = [func(x,fita[0],fita[1],fita[2],fita[3]) for x in new_xlist]

    if draw == True:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes.scatter(data_x, data_y*1e4,s = 10,c='black')
        #axes.scatter(new_xlist, new_ylist,s = 10,c='red')
        axes.set_xlabel("x", fontsize=15)
        axes.set_ylabel("y", fontsize=15)
        axes.legend()
        fig.tight_layout()

        plt.show()

    return new_ylist

def W_fig(draw):
    interval = 0.1
    def func(x,A,sigma,mu,B):
        return A*(1/np.sqrt(2*np.pi)/sigma)*np.exp(-  (x-mu)**2   /2/sigma**2    ) + B

    #fita, fitb = optimize.curve_fit(func, new_xlist, data_y, [-2235.5271988,-13.222923164,209,-59.7381196783])
    #new_ylist = [( func(x,fita[0],fita[1],fita[2],fita[3])*1e4 )**2/2 for x in new_xlist]
    new_yW = []
    for idx_datay in data_y:
        new_yW.append(idx_datay**2/2 *1e10)
    if draw == True:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes.scatter(data_x, data_y*1e4,s = 10,c='black')
        axes.scatter(data_x,new_yW,s = 3,c='red')
        axes.set_xlabel("x", fontsize=15)
        axes.set_ylabel("y", fontsize=15)
        axes.legend()
        fig.tight_layout()
        plt.show()
        return

def sum_allpoints(list):
    sum =0
    for ele in list:
        sum = sum + ele
    return sum



# print(area_raw(data_x, fitting_fig(False)))
#
# print(area_raw(data_x,data_y))


# factor = sum_allpoints(data_y)/len(data_y)
# print(factor)
# V = 0.7746/44*np.exp(factor)
# print( 'V=',V)
#print( 'Sum=',sum_allpoints(fitting_fig(True)))


# con_factor = V0 / area_raw(data_x,data_y)
# print('con_factor=',con_factor)
#fitting_fig(True)
W_fig(True)











# import time to name the file
import datetime
nowTime=datetime.datetime.now().strftime('%Y%m%d %H%M%S') #现在时间


'''Print on the txt '''
# create a new file with nowtime or delete 'nowtime' and open an existed file
filename = 'filefold/ex/ex' + nowTime + '.txt'

file1 = open(filename, 'w')
print('wordswords', file = file1)
