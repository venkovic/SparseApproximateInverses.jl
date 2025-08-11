import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

plt.rc('text.latex', preamble=r'\usepackage{amssymb} \usepackage{amsmath}')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

matrix = "4bw100eigs20k2"

fontsize = 10.5

res_cg = np.load("data/Experiment04_" + matrix + "_cg-res.npz")  
res_pmr = np.load("data/Experiment04_" + matrix + "_pcg-res_pmr.npz")
res_pcr = np.load("data/Experiment04_" + matrix + "_pcg-res_pcr.npz")
res_lopmr = np.load("data/Experiment04_" + matrix + "_pcg-res_lopmr.npz")

iters_cg = len(res_cg)
iters_pmr = min(len(res_pmr), 2 * iters_cg)
iters_pcr = min(len(res_pcr), 2 * iters_cg)
iters_lopmr = min(len(res_lopmr), 2 * iters_cg)

fig, ax = plt.subplots(figsize=(5.5, 3))

ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.semilogy(np.arange(0, iters_cg), res_cg[:iters_cg], 'r-', label='CG', linewidth=2)
ax.semilogy(np.arange(0, iters_pmr), res_pmr[:iters_pmr], color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), label='PCG (PMR)', linewidth=2)
ax.semilogy(np.arange(0, iters_pcr), res_pcr[:iters_pcr], 'k:', label='PCG (PCR)', linewidth=2)
ax.semilogy(np.arange(0, iters_lopmr), res_lopmr[:iters_lopmr], 'k-', label='PCG (LOPMR)', linewidth=2)

ax.set_xlabel('Iteration, ' + r'$i$', fontsize=fontsize)
ax.set_ylabel('Backward error, ' + r'$\|r_i\|_2/\|b\|_2$', fontsize=fontsize)
ax.set_title(matrix, fontsize=fontsize)
ax.legend(fontsize=fontsize)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=fontsize)

plt.tight_layout()
plt.savefig("img/Experiment04_" + matrix + "_pcg-res_norms.png", bbox_inches='tight', dpi=300)