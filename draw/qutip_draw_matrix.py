from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy import integrate
from scipy.integrate import quad, dblquad, nquad
import math
import csv
import matplotlib
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from colour import Color


def transform_qobj(qobj):
    new = []
    for ele in qobj:
        new_i = []
        for ele_i in ele[0]:
            new_i.append(ele_i)
        new.append(new_i)
    return new


def qobj_normal(qobj):
    return Qobj(transform_qobj(qobj))


def matrix_histogram(M, xlabels=None, ylabels=None, title=None, limits=None,
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

    def make_bar(ax, x0=0, y0=0, width=0.5, height=1, cmap="viridis",
                 norm=matplotlib.colors.Normalize(vmin=0, vmax=1), **kwargs):
        # Make data
        u = np.linspace(0, 2 * np.pi, 4 + 1) + np.pi / 4.
        v_ = np.linspace(np.pi / 4., 3. / 4 * np.pi, 100)
        v = np.linspace(0, np.pi, len(v_) + 2)
        v[0] = 0;
        v[-1] = np.pi;
        v[1:-1] = v_
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        xthr = np.sin(np.pi / 4.) ** 2;
        zthr = np.sin(np.pi / 4.)
        x[x > xthr] = xthr;
        x[x < -xthr] = -xthr
        y[y > xthr] = xthr;
        y[y < -xthr] = -xthr
        z[z > zthr] = zthr;
        z[z < -zthr] = -zthr

        x *= 1. / xthr * width;
        y *= 1. / xthr * width
        z += zthr
        z *= height / (2. * zthr)
        # translate
        x += x0;
        y += y0
        # plot
        ax.plot_surface(x, y, z, cmap=cmap, norm=norm, **kwargs)

    def frame(ax, M):
        M = M * 0.8
        ### Frame
        n = np.size(M)
        xpos, ypos = np.meshgrid(np.arange(M.shape[0]), np.arange(M.shape[1]))

        xpos = xpos.T.flatten() -0.05 # + 0.5 #- 0.5
        ypos = ypos.T.flatten() -0.05 # + 0.1
        zpos = np.zeros(n)  # + 0.1
        dx = dy = 0.8  * np.ones(n)
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
        cmap = cm.get_cmap('Greys')  # Spectral jet 'RdBu'
        # cmap = cm.get_cmap('GnBu')
        colors = cmap(norm(np.ones_like(dz)*-1), alpha=0.5)

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, alpha=0.1, color=colors,edgecolor='black', linewidth=0.7,shade=True,zorder=1)
        ax.set_alpha(0.5)
        return

    # ax.set_facecolor('white')
    # ax.set_alpha(0.5)
    if isinstance(M, Qobj):
        # extract matrix data from Qobj
        M = M.full()

    n = np.size(M)
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        # change seeing angle
        ax = fig.add_subplot(1, 1, 1, projection='3d', azim=-36, elev=36)
    # ax = Axes3D(fig, azim=-36, elev=36)

    xpos, ypos = np.meshgrid(np.arange(M.shape[0]), np.arange(M.shape[1]))

    xpos = xpos.T.flatten()  # + 0.5 #- 0.5
    ypos = ypos.T.flatten()  # + 0.1
    zpos = np.zeros(n)  # + 0.1
    dx = dy = 0.7 * np.ones(n)
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

    #norm = mpl.colors.Normalize(z_min, z_max)
    norm = mpl.colors.Normalize(-1, 1)
    cmap = cm.get_cmap('RdBu')  # Spectral jet 'RdBu'
    # cmap = cm.get_cmap('GnBu')
    colors = cmap(norm(dz), alpha=1)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, alpha=1, color=colors, edgecolor='black', linewidth=0.4, shade=True)
    #frame(ax=ax, M=M)
    if title and fig:
        ax.set_title(title)

    # '''Here change the bar location'''
    # x axis
    ax.axes.w_xaxis.set_major_locator(plt.IndexLocator(1, 0.35))
    if xlabels:
        # ticksx = np.arange(M.shape[0])
        # plt.xticks(ticksx, xlabels)

        ax.set_xticklabels(xlabels)
    ax.tick_params(axis='x', labelsize=10 ,rotation=45)

    # y axis
    ax.axes.w_yaxis.set_major_locator(plt.IndexLocator(1, 0.35))
    if ylabels:
        # ticksy = np.arange(M.shape[1])
        # plt.yticks(ticksy, ylabels)
        ax.set_yticklabels(ylabels,rotation=45)
    ax.tick_params(axis='y', labelsize=10,rotation=-45)

    # z axis
    ax.axes.w_zaxis.set_major_locator(plt.IndexLocator(0.5, -1))
    #ax.set_zlim3d([min(z_min, 0), z_max])
    ax.set_zlim3d([-1, 1])

    # ax.set_title('test')

    # color axis
    if colorbar:
        cax, kw = mpl.colorbar.make_axes(ax, shrink=.75, pad=.1)
        mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

    return fig, ax


def matrix_gradient(M, xlabels=None, ylabels=None, title=None, limits=None,
                    colorbar=True, fig=None, ax=None):
    def make_bar(ax, x0=0, y0=0, width=0.5, height=1, cmap="jet",
                 norm=matplotlib.colors.Normalize(vmin=-1, vmax=1), **kwargs):
        # Make data
        u = np.linspace(0, 2 * np.pi, 4 + 1) + np.pi / 4.
        v_ = np.linspace(np.pi / 4., 3. / 4 * np.pi, 100)
        v = np.linspace(0, np.pi, len(v_) + 2)
        v[0] = 0;
        v[-1] = np.pi;
        v[1:-1] = v_
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        xthr = np.sin(np.pi / 4.) ** 2;
        zthr = np.sin(np.pi / 4.)
        x[x > xthr] = xthr;
        x[x < -xthr] = -xthr
        y[y > xthr] = xthr;
        y[y < -xthr] = -xthr
        z[z > zthr] = zthr;
        z[z < -zthr] = -zthr

        x *= 1. / xthr * width;
        y *= 1. / xthr * width
        z += zthr
        z *= height / (2. * zthr)
        # translate
        x += x0
        y += y0
        # plot
        ax.plot_surface(x, y, z, cmap=cmap, norm=norm, **kwargs)

        # xpos = [x0 - width,x0 - width,x0 + width,x0 + width]
        # xpos = xpos - np.ones_like(xpos) * 0.15
        # ypos = [y0 - width , y0 + width, y0 - width , y0 + width ]
        # ypos = ypos - np.ones_like(ypos) * 0.15
        # zpos = np.zeros(4)
        # ones_type = np.ones_like(np.array(xpos))
        # ax.bar3d(xpos, ypos, zpos, dx = ones_type*0.1, dy= ones_type*0.1, dz = ones_type*height, alpha=1, color='black')

    def frame(ax, M):
        M = M
        ### Frame
        n = np.size(M)
        xpos, ypos = np.meshgrid(np.arange(M.shape[0]), np.arange(M.shape[1]))

        xpos = xpos.T.flatten() - 0.3  # + 0.5 #- 0.5
        ypos = ypos.T.flatten() - 0.3  # + 0.1
        zpos = np.zeros(n)  # + 0.1
        dx = dy = 0.6 * np.ones(n)
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
        cmap = cm.get_cmap('jet')  # Spectral jet 'RdBu'
        # cmap = cm.get_cmap('GnBu')
        colors = cmap(norm(dz), alpha=0.2)

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, alpha=1, color=colors, edgecolor='black', linewidth=0.8)
        return

    def make_bars(ax, x, y, height, width=0.3):
        widths = np.array(width) * np.ones_like(x)
        x = np.array(x).flatten()
        y = np.array(y).flatten()

        h = np.array(height).flatten()
        w = np.array(widths).flatten()
        norm = matplotlib.colors.Normalize(vmin=h.min(), vmax=h.max())
        for i in range(len(x.flatten())):
            make_bar(ax, x0=x[i], y0=y[i], width=w[i], height=h[i], norm=norm)

    # ax.set_facecolor('white')
    # ax.set_alpha(0.5)
    if isinstance(M, Qobj):
        # extract matrix data from Qobj
        M = M.full()

    n = np.size(M)
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        # change seeing angle
        ax = fig.add_subplot(1, 1, 1, projection='3d', azim=-36, elev=36)
    # ax = Axes3D(fig, azim=-36, elev=36)

    xpos, ypos = np.meshgrid(np.arange(M.shape[0]), np.arange(M.shape[1]))

    xpos = xpos.T.flatten()  # + 0.5 #- 0.5
    ypos = ypos.T.flatten()  # + 0.1
    zpos = np.zeros(n)  # + 0.1
    dx = dy = 0.6 * np.ones(n)
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

    #norm = mpl.colors.Normalize(z_min, z_max)
    norm = mpl.colors.Normalize(-1, 1)
    cmap = cm.get_cmap('jet')  # Spectral jet 'RdBu'
    # cmap = cm.get_cmap('GnBu')
    colors = cmap(norm(dz), alpha=0.9)

    make_bars(ax, x=xpos, y=ypos, height=dz, width=0.3)
    # frame(ax=ax, M=M)

    if title and fig:
        ax.set_title(title)

    # '''Here change the bar location'''
    # x axis
    ax.axes.w_xaxis.set_major_locator(plt.IndexLocator(1, 0.3))
    if xlabels:
        # ticksx = np.arange(M.shape[0])
        # plt.xticks(ticksx, xlabels)

        ax.set_xticklabels(xlabels)
    ax.tick_params(axis='x', labelsize=10,rotation=45)

    # y axis
    ax.axes.w_yaxis.set_major_locator(plt.IndexLocator(1, 0.3))
    if ylabels:
        # ticksy = np.arange(M.shape[1])
        # plt.yticks(ticksy, ylabels)
        ax.set_yticklabels(ylabels)  # ,rotation=-90)
    ax.tick_params(axis='y', labelsize=10,rotation=-45)

    # z axis
    ax.axes.w_zaxis.set_major_locator(plt.IndexLocator(0.5, -1))
    #ax.set_zlim3d([min(z_min, 0), z_max])
    ax.set_zlim3d([-1, 1])

    # ax.set_title('test')

    # color axis
    if colorbar:
        cax, kw = mpl.colorbar.make_axes(ax, shrink=.75, pad=.1)
        mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

    return fig, ax


def matrix_histogram_complex(M, xlabels=None, ylabels=None,
                             title=None, limits=None, phase_limits=None,
                             colorbar=True, fig=None, ax=None,
                             threshold=None):
    def complex_phase_cmap():
        cdict = {'blue': ((0.00, 0.0, 0.0),
                          (0.25, 0.0, 0.0),
                          (0.50, 1.0, 1.0),
                          (0.75, 1.0, 1.0),
                          (1.00, 0.0, 0.0)),
                 'green': ((0.00, 0.0, 0.0),
                           (0.25, 1.0, 1.0),
                           (0.50, 0.0, 0.0),
                           (0.75, 1.0, 1.0),
                           (1.00, 0.0, 0.0)),
                 'red': ((0.00, 1.0, 1.0),
                         (0.25, 0.5, 0.5),
                         (0.50, 0.0, 0.0),
                         (0.75, 0.0, 0.0),
                         (1.00, 1.0, 1.0))}

        cmap = mpl.colors.LinearSegmentedColormap('phase_colormap', cdict, 256)
        return cmap

    if isinstance(M, Qobj):
        # extract matrix data from Qobj
        M = M.full()

    n = np.size(M)
    xpos, ypos = np.meshgrid(range(M.shape[0]), range(M.shape[1]))
    xpos = xpos.T.flatten() - 0.5
    ypos = ypos.T.flatten() - 0.5
    zpos = np.zeros(n)
    dx = dy = 0.3 * np.ones(n)
    Mvec = M.flatten()
    dz = abs(Mvec)

    # make small numbers real, to avoid random colors
    idx, = np.where(abs(Mvec) < 0.001)
    Mvec[idx] = abs(Mvec[idx])

    if phase_limits:  # check that limits is a list type
        phase_min = phase_limits[0]
        phase_max = phase_limits[1]
    else:
        phase_min = -pi
        phase_max = pi

    norm = mpl.colors.Normalize(phase_min, phase_max)
    cmap = complex_phase_cmap()

    colors = cmap(norm(angle(Mvec)))
    if threshold is not None:
        colors[:, 3] = 1 * (dz > threshold)

    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig, azim=-35, elev=35)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

    if title and fig:
        ax.set_title(title)

    # x axis
    ax.axes.w_xaxis.set_major_locator(plt.IndexLocator(1, -0.5))
    if xlabels:
        ax.set_xticklabels(xlabels)
    ax.tick_params(axis='x', labelsize=12)

    # y axis
    ax.axes.w_yaxis.set_major_locator(plt.IndexLocator(1, -0.5))
    if ylabels:
        ax.set_yticklabels(ylabels)
    ax.tick_params(axis='y', labelsize=12)

    # z axis
    if limits and isinstance(limits, list):
        ax.set_zlim3d(limits)
    else:
        ax.set_zlim3d([0, 1])  # use min/max
    # ax.set_zlabel('abs')

    # color axis
    if colorbar:
        cax, kw = mpl.colorbar.make_axes(ax, shrink=.75, pad=.0)
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
        cb.set_ticks([-pi, -pi / 2, 0, pi / 2, pi])
        cb.set_ticklabels(
            (r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'))
        cb.set_label('arg')

    return fig, ax


test0 = Qobj([[0.2, 0.3, 0, 1, 0.3, 0, 1,0],
             [0.8, 0, 0, 0.2, 0.3, 0, 0,0],
              [0.8, 0, 0, 0.2, 0.3, 0, 0, 1],
             [0.1, 0, -0.7, 0.5, 0.3, 0, 0,0],
             [0.1, 0, -0.7, 0.5, 0.3, 0, 0,0],
              [0.1, 0, -0.7, 0.5, 0.3, 0, 0, 0],
              [0.1, 0, -0.7, 0.5, 0.3, 0, 0, 0],
             [0.4, 0.1, 0.1, 0.8, 0.3, 0, 1,0]
             ])

# yyyyy
# x
# x
# x
# matrix_histogram(test,xlabels=[r'$\left|00\right>$',r'$\left|01\right>$',r'$\left|10\right>$',r'$\left|11\right>$'],
# 				 ylabels=[r'$\left<00\right|$', r'$\left<01\right|$',r'$\left<10\right|$',r'$\left<11\right|$'])
ylabels=[r'$\left|000\right>$', r'$\left|001\right>$', r'$\left|010\right>$', r'$\left|011\right>$',
         r'$\left|100\right>$', r'$\left|101\right>$', r'$\left|110\right>$', r'$\left|111\right>$']
xlabels=[r'$\left<000\right|$', r'$\left<001\right>$', r'$\left<010\right|$', r'$\left<011\right|$',
         r'$\left<100\right|$', r'$\left<101\right>$', r'$\left<110\right|$', r'$\left<111\right|$']
fig = plt.figure(figsize=(14, 5))
for idx in range(2):
    test = rand_herm(8)
    ax = fig.add_subplot(1, 2, idx + 1, projection='3d', azim=-36, elev=36)
    matrix_gradient(test0, ylabels=ylabels,xlabels=xlabels,ax=ax)

fig = plt.figure(figsize=(14, 6))
for idx in range(2):
    test = rand_herm(8)
    ax = fig.add_subplot(1, 2, idx + 1, projection='3d', azim=-36, elev=36)
    matrix_histogram(test, ylabels=ylabels, xlabels=xlabels, ax=ax)
plt.show()
