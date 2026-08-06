import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread
import matplotlib.gridspec as gridspec

plt.rc('text.latex', preamble=r'\usepackage{amssymb} \usepackage{amsmath}')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

matrix = "bcsstk21" # \in {"bcsstk21", "tri100eigs4k", "msc04515"}
fontsize = 10.5

R_norm_pmr = np.load("data/Experiment02_" + matrix + "_backward_error_pmr_spd.npz")
R_norm_pcg = np.load("data/Experiment02_" + matrix + "_backward_error_pcg.npz")
R_norm_lopmr = np.load("data/Experiment02_" + matrix + "_backward_error_lopmr_spd.npz")

iters_pmr = len(R_norm_pmr)
iters_pcg = len(R_norm_pcg)
iters_lopmr = len(R_norm_lopmr)

fig, ax = plt.subplots(figsize=(5.5, 3))

ax.semilogy(np.arange(0, iters_pmr), R_norm_pmr, color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), label='PMR_SPD', linewidth=2)
ax.semilogy(np.arange(0, iters_pcg), R_norm_pcg, 'k:', label='PCG', linewidth=2)
ax.semilogy(np.arange(0, iters_lopmr), R_norm_lopmr, 'k-', label='LOPMR_SPD', linewidth=2)

ax.set_xlabel('Iteration, ' + r'$i$', fontsize=fontsize)
ax.set_ylabel('Backward error, ' + r'$\eta_A(M_i^{-1})$', fontsize=fontsize)
ax.set_title(matrix, fontsize=fontsize)
ax.legend(fontsize=fontsize)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=fontsize)

plt.tight_layout()
plt.savefig("img/Experiment02_" + matrix + "_backward_error.png", bbox_inches='tight', dpi=300)