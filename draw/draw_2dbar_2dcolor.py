

"""
===========================
Frontpage histogram example
===========================

This example reproduces the frontpage histogram example.
"""

import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import xlrd
import csv
from qutip import *


def matrix_histogram_tao(M, xlabels=None, ylabels=None, title=None, limits=None,
                     colorbar=True, fig=None, ax=None):
    """
    Draw a histogram for the matrix M, with the given x and y labels and title.

    Parameters
    ----------
    M : Matrix of Qobj
        The matrix to visualize

    xlabels : list of strings
        list of x labels

    ylabels : list of strings
        list of y labels

    title : string
        title of the plot (optional)

    limits : list/array with two float numbers
        The z-axis limits [min, max] (optional)

    ax : a matplotlib axes instance
        The axes context in which the plot will be drawn.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.

    Raises
    ------
    ValueError
        Input argument is not valid.

    """

    if isinstance(M, Qobj):
        # extract matrix data from Qobj
        M = M.full()

    n = np.size(M)
    xpos, ypos = np.meshgrid(np.arange(1,M.shape[0]+1,1), np.arange(1,M.shape[1]+1,1))
    xpos = xpos.T.flatten() - 0.5
    ypos = ypos.T.flatten() - 0.5
    zpos = np.zeros(n)
    dx = dy = 0.25 * np.ones(n)
    dz = np.real(M.flatten())

    if limits and type(limits) is list and len(limits) == 2:
        z_min = limits[0]
        z_max = limits[1]
    else:
        z_min = min(dz)
        z_max = max(dz)
        if z_min == z_max:
            z_min -= 0.1
            z_max += 0.1

    norm = mpl.colors.Normalize(z_min, z_max)
    # cmap = cm.get_cmap('jet')  # Spectral
    # cmap = plt.get_cmap('bwr')
    #cmap = plt.get_cmap('GnBu')
    cmap = plt.get_cmap('RdBu')
    colors = cmap(norm(dz))

    if ax is None:
        fig = plt.figure(figsize=(10,8))
        ax = Axes3D(fig, azim=-35, elev=35)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

    if title and fig:
        ax.set_title(title)

    # x axis
    ax.axes.w_xaxis.set_major_locator(plt.IndexLocator(1, -0.5))
    if xlabels:
        ax.set_xticklabels(xlabels)
    ax.tick_params(axis='x', labelsize=14)

    # y axis
    ax.axes.w_yaxis.set_major_locator(plt.IndexLocator(1, -0.5))
    if ylabels:
        ax.set_yticklabels(ylabels)
    ax.tick_params(axis='y', labelsize=14)

    # z axis
    ax.axes.w_zaxis.set_major_locator(plt.IndexLocator(1, 0.5))
    ax.set_zlim3d([min(z_min, 0), z_max])

    # color axis
    if colorbar:
        cax, kw = mpl.colorbar.make_axes(ax, shrink=.75, pad=.0)
        mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

    return fig, ax


def draw_picauto():

    ifile = open('mat_corrected.csv', "r")
    # reader is a list [[row1],[row2],...]
    reader = csv.reader(ifile)
    col2 = []
    col3 = []
    col4 = []
    for idx, row in enumerate(reader):
        col2.append(float(row[2]))
        col3.append(float(row[3]))
        col4.append(float(row[4]))

    ifile = open('algo_corrected.csv', "r")
    # reader is a list [[row1],[row2],...]
    reader = csv.reader(ifile)
    col6 = []
    for idx, row in enumerate(reader):
        col6.append(float(row[3]))

    # xl=xlrd.open_workbook(r'full01.xlsx')
    # sheet=xl.sheet_by_index(0)
    #
    # col2 = sheet.col_values(2)
    # col3 = sheet.col_values(3)
    # col4 = sheet.col_values(4)
    # col6 = sheet.col_values(6)


    z = []
    z_th = []
    fid = []
    acc = []
    acc_ideal = []


    for m in range(1,max,1):
        if m == 0:
            continue
        z0 = []
        z_th0 = []
        fid0 = []
        acc0 = []
        acc_ideal0 = []
        for n in range(1,max,1):
            idx = max * m + n
            z0.append(col2[idx])
            z_th0.append(col3[idx])
            fid0.append(col4[idx])
            acc0.append(col6[idx])
            if m==n:
                acc_ideal0.append(1)
            else:
                acc_ideal0.append(0)
        z0.append(1)
        z_th0.append(1)
        fid0.append(1)
        acc0.append(1)
        acc_ideal0.append(1)
        z.append(z0)
        z_th.append(z_th0)
        fid.append(fid0)
        acc.append(acc0)
        acc_ideal.append(acc_ideal0)
    # occupy the place of 0
    # count from left so add one more
    z.append([0*x + 1 for x in range(max)])
    z_th.append([0*x + 1 for x in range(max)])
    fid.append([0*x + 1 for x in range(max)])
    acc.append([0*x + 1 for x in range(max)])
    acc_ideal.append([0*x + 1 for x in range(max)])

    # make these smaller to increase the resolution
    dx, dy = 1, 1

    # generate 2 2d grids for the x & y bounds
    y, x = np.mgrid[slice(1, max + dy, dy),
                    slice(1, max + dx, dx)]

    # z = np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)
    # print(z)

    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    # z = z[:-1, :-1]
    # levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())


    levels = MaxNLocator(nbins=200).tick_values(0, 1)
    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    #cmap = plt.get_cmap('bwr')
    cmap = plt.get_cmap('GnBu')
    #cmap = plt.get_cmap('RdBu')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    fig, ((ax0,ax1),(ax2,ax3)) = plt.subplots(nrows=2,ncols=2,figsize=(8, 8))

    im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax0)
    ax0.set_title(r'$P_0$ in Experiment')
    fig.tight_layout()

    im = ax1.pcolormesh(x, y, z_th, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax1)
    ax1.set_title(r'$P_0$ in Theory')
    fig.tight_layout()

    # fig2, (ax2,ax3) = plt.subplots(nrows=1,ncols=2,figsize=(8, 4))
    # cmap = plt.get_cmap('bwr')

    # im = ax2.pcolormesh(x, y, fid, cmap=cmap, norm=norm)
    # fig.colorbar(im, ax=ax2)
    # ax2.set_title(r'Fidelity')
    # fig.tight_layout()

    im = ax2.pcolormesh(x, y, acc, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax2)
    ax2.set_title(r'Experimental accepting rate')
    fig.tight_layout()

    im = ax3.pcolormesh(x, y, acc_ideal, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax3)
    ax3.set_title(r'The most ideal accepting rate')
    fig.tight_layout()

    plt.show()
    return
# draw_picauto()


def ibm_temp():
    nlist = np.arange(1,7,1)
    ibm = [0.925,0.969,0.942,0.941,0.929,0.939]
    we = [0.9115,0.9697,0.9139,0.9247,0.9137,0.916]
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    axes.plot(nlist, ibm, c='red', label='ibm')
    axes.scatter(nlist, we, c='blue', label='our')
    axes.set_xlabel(r'$P_0$')
    axes.set_ylabel('m (m=n)')
    axes.legend()
    fig.tight_layout()
    plt.show()


# ibm_temp()

def draw_3Dauto():
    ifile = open('mat_corrected.csv', "r")
    # reader is a list [[row1],[row2],...]
    reader = csv.reader(ifile)
    col2 = []
    col3 = []
    col4 = []
    for idx, row in enumerate(reader):
        col2.append(float(row[2]))
        col3.append(float(row[3]))
        col4.append(float(row[4]))

    ifile = open('algo_corrected.csv', "r")
    # reader is a list [[row1],[row2],...]
    reader = csv.reader(ifile)
    col6 = []
    for idx, row in enumerate(reader):
        col6.append(float(row[3]))

    # xl=xlrd.open_workbook(r'full01.xlsx')
    # sheet=xl.sheet_by_index(0)
    #
    # col2 = sheet.col_values(2)
    # col3 = sheet.col_values(3)
    # col4 = sheet.col_values(4)
    # col6 = sheet.col_values(6)


    z = []
    z_th = []
    fid = []
    acc = []
    acc_ideal = []


    for m in range(1,m_max+1,1):
        if m == 0:
            continue
        z0 = []
        z_th0 = []
        fid0 = []
        acc0 = []
        acc_ideal0 = []
        for n in range(1,n_max+1,1):
            idx = max * m + n
            z0.append(col2[idx])
            z_th0.append(col3[idx])
            fid0.append(col4[idx])
            acc0.append(col6[idx])
            if m==n:
                acc_ideal0.append(1)
            else:
                acc_ideal0.append(0)
        # z0.append(1)
        # z_th0.append(1)
        # fid0.append(1)
        # acc0.append(1)
        # acc_ideal0.append(1)
        z.append(z0)
        z_th.append(z_th0)
        fid.append(fid0)
        acc.append(acc0)
        acc_ideal.append(acc_ideal0)
    # occupy the place of 0
    # count from left so add one more
    # z.append([0*x  for x in range(max)])
    # z_th.append([0*x  for x in range(max)])
    # fid.append([0*x  for x in range(max)])
    # acc.append([0*x  for x in range(max)])
    # acc_ideal.append([0*x for x in range(max)])

    time = '181019'
    matrix_histogram_tao(Qobj(z ),title=r'$Exp\  P_0$', limits=[0,1])
    plt.savefig('output//figfinal//f1_'+time+'.pdf')

    matrix_histogram_tao(Qobj(z_th ), title=r'$Ideal\  P_0$', limits=[0,1])
    plt.savefig('output//figfinal//f2_' + time + '.pdf')

    matrix_histogram_tao(Qobj(acc_ideal ), title=r'$Ideal \ P_{acc}$', limits=[0,1])
    plt.savefig('output//figfinal//f3_' + time + '.pdf')

    matrix_histogram_tao(Qobj(acc ), title=r'$Exp \ P_{acc}$', limits=[0,1])
    plt.savefig('output//figfinal//f4_' + time + '.pdf')

    plt.show()
    # z = np.array(z)
    # xs = np.arange(max)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for y in (np.arange(max)):
    #     height = z[:, y]
    #     ax.bar(left=xs, height=height, zs=y, zdir='y', alpha=0.8)
    #
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    #
    # plt.show()

    return


max = 1+9 # 9+1#15+1
# for drawing
m_max, n_max = 6,6 # 6,6#15,15
draw_3Dauto()



