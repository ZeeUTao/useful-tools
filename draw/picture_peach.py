def set_figorder(ax,order = 'a'):
    ax.set_title(order, x=0.1, y=0.9,fontsize = 14)
    return	


def draw_2d(kx,ky,kz,fig=None,ax=None,fig_label=None,xlabel=None,ylabel=None):
	if fig == None:
		fig = plt.figure(figsize=(8,6))
		ax = fig.add_subplot(1, 1, 1)
	if ax == None:
		ax = fig.add_subplot(1, 1, 1)
	
	font_xylabel = 12
	cmap = plt.get_cmap('RdBu')

	levels = MaxNLocator(nbins=100).tick_values(kz.min(), kz.max())
	norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

	im = ax.pcolormesh(kx,ky,kz,cmap=cmap, norm=norm)
	cbar = fig.colorbar(im, ax=ax)
	#ax.legend()
	ax.set_title(fig_label)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_xticks(np.linspace(kx.min(), kx.max(), 4))
	ax.set_yticks(np.linspace(ky.min(), ky.max(), 4))
	return
	
def draw_surface(kx,ky,kz,fig=None,ax=None,fig_label=None,xlabel=None,ylabel=None,order = None
	):
	from matplotlib import cm
	if fig == None:
		fig = plt.figure(figsize=(8,6))
		ax = fig.add_subplot(1, 1, 1)
	if ax == None:
		ax = plt.subplot(111, projection='3d')
	
	font_xylabel = 12
	cmap = plt.get_cmap('inferno')
	#surf = ax.plot_wireframe(kx, ky, kz, rstride=1, cstride=1,  cmap=cm.coolwarm)
	#	plot_surface plot_trisurf RdBu inferno
	levels = MaxNLocator(nbins=100).tick_values(kz.min(), kz.max())
	norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

	surf = ax.plot_surface(kx[:-1], ky[:-1], kz[:-1],  cmap=cmap, linewidth=0, antialiased=False)
	cbar = fig.colorbar(surf, ax=ax,shrink=0.5, aspect=10)
	ax.set_xticks(np.linspace(kx.min(), kx.max(), 5))
	
	y_interval = (ky.max()+1)/4
	ax.set_yticks(np.arange(ky.min(), ky.max() + y_interval , y_interval))
	
	# ax.set_ylim(0, 16)
	
	
	set_figorder(ax,order = order)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)


	# Set viewpoint.
	ax.azim,ax.elev  = -120,30
	return 
