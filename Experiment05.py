import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread
import matplotlib.gridspec as gridspec

plt.rc('text.latex', preamble=r'\usepackage{amssymb} \usepackage{amsmath}')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

matrix = "rand20k" # \in {"bundle1", 
                    #      "4bw100eigs20k", 
                    #      "rand20k",
                    #      "rand20k2",
                    #      "wathen100", 
                    #      "Poisson32k"}

fontsize = 10.5

R_norm_pmr = np.load("data/Experiment05_" + matrix + "_R_norm_pmr_spai.npz")
R_norm_pcg = np.load("data/Experiment05_" + matrix + "_R_norm_pcg_spai.npz")
R_norm_lopmr = np.load("data/Experiment05_" + matrix + "_R_norm_lopmr_spai.npz")

iters_pmr = len(R_norm_pmr)
iters_pcg = len(R_norm_pcg)
iters_lopmr = len(R_norm_lopmr)

fig = plt.figure(figsize=(6.15  , 3.7))
gs = gridspec.GridSpec(3, 2, width_ratios=[2.5, .8], wspace=-.03, hspace=0.3)

ax_conv = fig.add_subplot(gs[:, 0])

ax_spy_pmr = fig.add_subplot(gs[0, 1])
ax_spy_pcg = fig.add_subplot(gs[1, 1])
ax_spy_lopmr = fig.add_subplot(gs[2, 1])

ax_conv.semilogy(np.arange(0, iters_pmr), R_norm_pmr, color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), label='PMR-SPAI', linewidth=2)
ax_conv.semilogy(np.arange(0, iters_pcg), R_norm_pcg, 'k:', label='PCG-SPAI', linewidth=2)
ax_conv.semilogy(np.arange(0, iters_lopmr), R_norm_lopmr, 'k-', label='LOPMR-SPAI', linewidth=2)

ax_conv.set_xlabel('Global SPAI iteration, ' + r'$i$', fontsize=fontsize)
ax_conv.set_ylabel('Residual norm, ' + r'$\|R_i\|_F$', fontsize=fontsize)
ax_conv.set_title(matrix, fontsize=fontsize)
ax_conv.legend(fontsize=fontsize)
ax_conv.grid(True, alpha=0.3)
ax_conv.tick_params(labelsize=fontsize)

M = mmread("data/Experiment05_" + matrix + "_M_pmr_spai.mtx")
if False:
  ax_spy_pmr.spy(M, markersize=0.05, color='black', marker='.')
else:
  coo = M.tocoo()
  ax_spy_pmr.scatter(coo.col, coo.row, s=.01, c='black', marker='o', 
             edgecolors='none', linewidths=0)
  ax_spy_pmr.invert_yaxis()
  ax_spy_pmr.set_aspect('equal')
ax_spy_pmr.set_xticks([])
ax_spy_pmr.set_yticks([])
ax_spy_pmr.set_title('PMR', fontsize=fontsize)

M = mmread("data/Experiment05_" + matrix + "_M_pcg_spai.mtx")
if False:
  ax_spy_pcg.spy(M, markersize=0.05, color='black', marker='.')
else:
  coo = M.tocoo()
  ax_spy_pcg.scatter(coo.col, coo.row, s=.01, c='black', marker='o', 
             edgecolors='none', linewidths=0)
  ax_spy_pcg.invert_yaxis()
  ax_spy_pcg.set_aspect('equal')
ax_spy_pcg.set_xticks([])
ax_spy_pcg.set_yticks([])
ax_spy_pcg.set_title('PCG', fontsize=fontsize)

M = mmread("data/Experiment05_" + matrix + "_M_lopmr_spai.mtx")
if False:
  ax_spy_lopmr.spy(M, markersize=0.05, color='black', marker='.')
else:
  coo = M.tocoo()
  ax_spy_lopmr.scatter(coo.col, coo.row, s=.01, c='black', marker='o', 
             edgecolors='none', linewidths=0)
  ax_spy_lopmr.invert_yaxis()
  ax_spy_lopmr.set_aspect('equal')
ax_spy_lopmr.set_xticks([])
ax_spy_lopmr.set_yticks([])
ax_spy_lopmr.set_title('LOPMR', fontsize=fontsize)

plt.tight_layout()
plt.savefig("img/Experiment05_" + matrix + ".png", bbox_inches='tight', dpi=300)