

"""
===========================
Frontpage histogram example
===========================

This example reproduces the frontpage histogram example.
"""


import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import xlrd

xl=xlrd.open_workbook(r'full_15.xlsx')
sheet=xl.sheet_by_index(0)

col2 = sheet.col_values(2)
col3 = sheet.col_values(3)
col4 = sheet.col_values(4)
col6 = sheet.col_values(6)
z = []
z_th = []
fid = []
acc = []

max = 15+1
for m in range(1,max,1):
    if m == 0:
        continue
    z0 = []
    z_th0 = []
    fid0 = []
    acc0 = []
    for n in range(1,max,1):
        idx = max * m + n
        z0.append(col2[idx])
        z_th0.append(col3[idx])
        fid0.append(col4[idx])
        acc0.append(col6[idx])

    z0.append(1)
    z_th0.append(1)
    fid0.append(1)
    acc0.append(1)
    z.append(z0)
    z_th.append(z_th0)
    fid.append(fid0)
    acc.append(acc0)
# count from left so add one more
z.append([0*x + 1 for x in range(max)])
z_th.append([0*x + 1 for x in range(max)])
fid.append([0*x + 1 for x in range(max)])
acc.append([0*x + 1 for x in range(max)])

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
# cmap = plt.get_cmap('bwr')
cmap = plt.get_cmap('RdBu')
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
cmap = plt.get_cmap('bwr')
im = ax2.pcolormesh(x, y, fid, cmap=cmap, norm=norm)
fig.colorbar(im, ax=ax2)
ax2.set_title(r'Fidelity')
fig.tight_layout()

cmap = plt.get_cmap('bwr')
im = ax3.pcolormesh(x, y, acc, cmap=cmap, norm=norm)
fig.colorbar(im, ax=ax3)
ax3.set_title(r'Accuracy of 2qcfa')
fig.tight_layout()

plt.show()













%------------------------------PS


def draw_bias2d(ax=None,fig=None,font_xylabel = 12):
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    if fig == None and ax == None:
        fig = plt.figure(figsize=(8,6))
    if ax == None:
        ax = fig.add_subplot(1, 1, 1)

    ifile = open(power2d, "r")
    # reader is a list [[row1],[row2],...]
    reader = csv.reader(ifile)

    freq_step = 26

    freq,power,amp = [],[],[]
    x,y,z = [],[],[]
    for idx, row in enumerate(reader):
        x.append(float(row[0]))
        y.append(float(row[1]))
        z.append(float(row[3]))
        if (idx+1) % freq_step == 0 and idx > 0:
            if (idx+1) == freq_step:
                freq, power, amp = np.array([x]),np.array([y]),np.array([z])
            else:
                freq = np.concatenate((freq, np.array([x])), axis=0)
                power = np.concatenate((power, np.array([y])), axis=0)
                amp = np.concatenate((amp, np.array([z])), axis=0)
            x, y, z = [], [], []


    levels = MaxNLocator(nbins=1000).tick_values(amp.min(), amp.max())

    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.get_cmap('RdBu')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    im = ax.pcolormesh(freq,power, amp, cmap=cmap, norm=norm)
    #im = ax.contourf(freq, power, amp, cmap=cmap, norm=norm)

    cbar = fig.colorbar(im, ax=ax)
    bar_scale = 1000
    cbar.set_ticks(np.linspace(amp.min()//bar_scale *bar_scale + bar_scale, amp.max()//bar_scale*bar_scale, 5))
    #ax.set_title('pcolormesh with levels')
    ax.set_xlabel("Frequency(GHz)", fontsize=font_xylabel)
    ax.set_ylabel(r"$\left|S_{21}\right|$(a.u.)", fontsize=font_xylabel)
    ax.set_xticks(np.linspace(freq.min(), freq.max(), 5))
    ax.set_yticks(np.linspace(power.min(), power.max(), 6))
    ax.legend()
    fig.tight_layout()
    return
