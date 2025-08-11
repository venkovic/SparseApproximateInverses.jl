import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread
import matplotlib.gridspec as gridspec

plt.rc('text.latex', preamble=r'\usepackage{amssymb} \usepackage{amsmath}')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

matrix = "msc04515" # \in {"bcsstk21", "tri100eigs4k", "msc04515"}
fontsize = 10.5

R_norm_psd = np.load("data/Experiment01_" + matrix + "_R_norm_psd.npz")
R_norm_pcg = np.load("data/Experiment01_" + matrix + "_R_norm_pcg.npz")
R_norm_pmr = np.load("data/Experiment01_" + matrix + "_R_norm_pmr.npz")
R_norm_pcr = np.load("data/Experiment01_" + matrix + "_R_norm_pcr.npz")
R_norm_lopmr = np.load("data/Experiment01_" + matrix + "_R_norm_lopmr.npz")

iters_psd = len(R_norm_psd)
iters_pcg = len(R_norm_pcg)
iters_pmr = len(R_norm_pmr)
iters_pcr = len(R_norm_pcr)
iters_lopmr = len(R_norm_lopmr)

fig, ax = plt.subplots(figsize=(5.5, 3))

ax.semilogy(np.arange(0, iters_psd), R_norm_psd, 'k--', label='PSD', linewidth=2)
ax.semilogy(np.arange(0, iters_pcg), R_norm_pcg, 'k-.', label='PCG', linewidth=2)
ax.semilogy(np.arange(0, iters_pmr), R_norm_pmr, color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), label='PMR', linewidth=2)
ax.semilogy(np.arange(0, iters_pcr), R_norm_pcr, 'k:', label='PCR', linewidth=2)
ax.semilogy(np.arange(0, iters_lopmr), R_norm_lopmr, 'k-', label='LOPMR', linewidth=2)

ax.set_xlabel('Iteration, ' + r'$i$', fontsize=fontsize)
ax.set_ylabel('Residual norm, ' + r'$\|R_i\|_F$', fontsize=fontsize)
ax.set_title(matrix, fontsize=fontsize)
ax.legend(fontsize=fontsize)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=fontsize)

plt.tight_layout()
plt.savefig("img/Experiment01_" + matrix + "_R_norms.png", bbox_inches='tight', dpi=300)