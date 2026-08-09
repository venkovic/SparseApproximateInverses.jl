import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

plt.rc('text.latex', preamble=r'\usepackage{amssymb} \usepackage{amsmath}')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

matrix = "bundle1"        # \in {"bundle1", 
                          #      "4bw100eigs20k", 
                          #      "rand20k"
                          #      "rand20k2"
                          #      "wathen100", 
                          #      "Poisson32k"}
fontsize = 10.5
nreals = 20

res_cg = [np.load("data/Experiment06_" + matrix + "_cg-res_real%d.npz" % (ireal + 1)) for ireal in range(nreals)]
res_pmr = [np.load("data/Experiment06_" + matrix + "_pcg-res_pmr_spai_real%d.npz" % (ireal + 1)) for ireal in range(nreals)]
res_pcg = [np.load("data/Experiment06_" + matrix + "_pcg-res_pcg_spai_real%d.npz" % (ireal + 1)) for ireal in range(nreals)]
res_lopmr = [np.load("data/Experiment06_" + matrix + "_pcg-res_lopmr_spai_real%d.npz" % (ireal + 1)) for ireal in range(nreals)]

#iters_cg = len(res_cg)
#iters_pmr = min(len(res_pmr), 2 * iters_cg)
#iters_pcg = min(len(res_pcg), 2 * iters_cg)
#iters_lopmr = min(len(res_lopmr), 2 * iters_cg)

fig, ax = plt.subplots(figsize=(5.5, 3))

ax.xaxis.set_major_locator(MaxNLocator(integer=True))

#ax.semilogy(np.arange(0, iters_cg), res_cg[:iters_cg], 'r-', label='CG', linewidth=2)
#ax.semilogy(np.arange(0, iters_pmr), res_pmr[:iters_pmr], color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), label='PCG (PMR-SPAI)', linewidth=2)
#ax.semilogy(np.arange(0, iters_pcg), res_pcg[:iters_pcg], 'k:', label='PCG (PCG-SPAI)', linewidth=2)
#ax.semilogy(np.arange(0, iters_lopmr), res_lopmr[:iters_lopmr], 'k-', label='PCG (LOPMR-SPAI)', linewidth=2)
#ax.semilogy(np.arange(0, iters_pmr), res_pmr[:iters_pmr], color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), label='PCG (MR)', linewidth=2)
#ax.semilogy(np.arange(0, iters_pcg), res_pcg[:iters_pcg], 'k:', label='PCG (CG)', linewidth=2)
#ax.semilogy(np.arange(0, iters_lopmr), res_lopmr[:iters_lopmr], 'k-', label='PCG (LOMR)', linewidth=2)

lw = .5

ax.semilogy(np.arange(0, len(res_cg[0])), res_cg[0], 'r-', label='pcg (Jacobi)', linewidth=lw)
ax.semilogy(np.arange(0, len(res_pmr[0])), res_pmr[0], 'c-', label='pcg (PMR_SPD SPAI)', linewidth=lw)
ax.semilogy(np.arange(0, len(res_pcg[0])), res_pcg[0], 'k-', label='pcg (PCG SPAI)', linewidth=lw)
ax.semilogy(np.arange(0, len(res_lopmr[0])), res_lopmr[0], 'g-', label='pcg (LOPMR_SPD SPAI)', linewidth=lw)

for ireal in range(1, nreals):
  ax.semilogy(np.arange(0, len(res_cg[ireal])), res_cg[ireal], 'r-', linewidth=lw)
  ax.semilogy(np.arange(0, len(res_pmr[ireal])), res_pmr[ireal], 'c-', linewidth=lw)
  ax.semilogy(np.arange(0, len(res_pcg[ireal])), res_pcg[ireal], 'k-', linewidth=lw)
  ax.semilogy(np.arange(0, len(res_lopmr[ireal])), res_lopmr[ireal], 'g-', linewidth=lw)

ax.set_xlabel('PCG iteration, ' + r'$i$', fontsize=fontsize)
ax.set_ylabel('Backward error, ' + r'$\|r_i\|_2/\|b\|_2$', fontsize=fontsize)
ax.set_title(matrix)
#ax.set_title('Linear solve preconditioned by approximate inverse', fontsize=fontsize)
ax.legend(fontsize=fontsize)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=fontsize)

plt.tight_layout()
plt.savefig("img/Experiment06_" + matrix + "_pcg-res_norms.png", bbox_inches='tight', dpi=300)
#plt.savefig("img/Experiment06_" + matrix + "_pcg-res_norms_SIAM_PP26.png", bbox_inches='tight', dpi=300)
