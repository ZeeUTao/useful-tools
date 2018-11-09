from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy import integrate
from scipy.integrate import quad,dblquad,nquad
import math
import csv
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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

    if isinstance(M, Qobj):
        # extract matrix data from Qobj
        M = M.full()

    n = np.size(M)
    xpos, ypos = np.meshgrid(range(M.shape[0]), range(M.shape[1]))
    xpos = xpos.T.flatten() - 0.5
    ypos = ypos.T.flatten() - 0.5
    zpos = np.zeros(n)
    dx = dy = 0.3 * np.ones(n)
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
    cmap = cm.get_cmap('RdBu')  # Spectral jet
    # cmap = cm.get_cmap('GnBu')
    colors = cmap(norm(dz))

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
