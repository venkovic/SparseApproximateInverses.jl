import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread
import matplotlib.gridspec as gridspec

plt.rc('text.latex', preamble=r'\usepackage{amssymb} \usepackage{amsmath}')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

matrix = "tri100eigs4k"  # \in {"bcsstk21", 
                      #      "tri100eigs4k", 
                      #      "triclust4k",
                      #      "triunif4k",
                      #      "msc04515"}

fontsize = 10.5

R_norm_psd = np.load("data/Experiment01_" + matrix + "_R_norm_psd.npz")
R_norm_npcg = np.load("data/Experiment01_" + matrix + "_R_norm_npcg.npz")
R_norm_pmr = np.load("data/Experiment01_" + matrix + "_R_norm_pmr.npz")
R_norm_pcg = np.load("data/Experiment01_" + matrix + "_R_norm_pcg.npz")
R_norm_lopmr = np.load("data/Experiment01_" + matrix + "_R_norm_lopmr.npz")

iters_psd = len(R_norm_psd)
iters_npcg = len(R_norm_npcg)
iters_pmr = len(R_norm_pmr)
iters_pcg = len(R_norm_pcg)
iters_lopmr = len(R_norm_lopmr)

fig, ax = plt.subplots(figsize=(5.5, 3))

ax.semilogy(np.arange(0, iters_psd), R_norm_psd, 'k--', label='PSD', linewidth=2)
ax.semilogy(np.arange(0, iters_npcg), R_norm_npcg, 'k-.', label='NPCG', linewidth=2)
ax.semilogy(np.arange(0, iters_pmr), R_norm_pmr, color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), label='PMR', linewidth=2)
ax.semilogy(np.arange(0, iters_pcg), R_norm_pcg, 'k:', label='PCG', linewidth=2)
ax.semilogy(np.arange(0, iters_lopmr), R_norm_lopmr, 'k-', label='LOPMR', linewidth=2)

ax.set_xlabel('Iteration, ' + r'$i$', fontsize=fontsize)
ax.set_ylabel('Residual norm, ' + r'$\|R_i\|_F$', fontsize=fontsize)
ax.set_title(matrix, fontsize=fontsize)
ax.legend(fontsize=fontsize)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=fontsize)

plt.tight_layout()
plt.savefig("img/Experiment01_" + matrix + "_R_norms.png", bbox_inches='tight', dpi=300)



fig, ax = plt.subplots(figsize=(5.5, 3))

ax.semilogy(np.arange(0, iters_psd), R_norm_psd, 'k--', label='SD', linewidth=2)
ax.semilogy(np.arange(0, iters_pmr), R_norm_pmr, color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), label='MR', linewidth=2)
ax.semilogy(np.arange(0, iters_pcg), R_norm_pcg, 'k:', label='CG', linewidth=2)
ax.semilogy(np.arange(0, iters_lopmr), R_norm_lopmr, 'k-', label='LOMR', linewidth=2)

ax.set_xlabel('Iteration, ' + r'$i$', fontsize=fontsize)
ax.set_ylabel('Residual norm, ' + r'$\|R_i\|_F$', fontsize=fontsize)
ax.set_title(matrix, fontsize=fontsize)
ax.legend(fontsize=fontsize)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=fontsize)

plt.tight_layout()
plt.savefig("img/Experiment01_" + matrix + "_R_norms_SIAM_PP26.png", bbox_inches='tight', dpi=300)