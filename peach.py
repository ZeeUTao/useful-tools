import numpy as np
from qutip import *
import qutip.tensor as t
import matplotlib.pyplot as plt
# from picture import *


x, y, z, i = sigmax(),sigmay(),sigmaz(),qeye(2)


def qpt(gate = sigmax()):
	q_num = gate.shape[0] / 2
	U_rho = spre(gate) * spost(gate.dag())
	op_basis = [[qeye(2), sigmax(), sigmay(), sigmaz()]] * q_num
	op_label = [["i", "x", "y", "z"]] * q_num
	chi = qutip.qpt(U_rho, op_basis)
	fig = qpt_plot_combined(chi, op_label)
	return Qobj(chi)
	
def draw_mat(mat,imag = False, complex = False):
	mat = (Qobj(mat))
	matrix_histogram(np.real(mat.full()))
	if imag: matrix_histogram(np.imag(mat.full()))
	if complex: matrix_histogram_complex(Qobj(mat))