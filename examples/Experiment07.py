import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

plt.rc('text.latex', preamble=r'\usepackage{amssymb} \usepackage{amsmath}')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

matrix = "Poisson32k"  # \in {"bundle1", 
                      #      "4bw100eigs20k", 
                      #      "4bw100eigs20k2", 
                      #      "rand20k",
                      #      "rand20k2",
                      #      "wathen100", 
                      #      "Poisson32k"}

fontsize = 10.5

pcg_iters_pmr = np.load("data/Experiment07_" + matrix + "_pcg_iters_pmr_spd_spai.npz")
pcg_iters_pcg = np.load("data/Experiment07_" + matrix + "_pcg_iters_pcg_spai.npz")
pcg_iters_lopmr = np.load("data/Experiment07_" + matrix + "_pcg_iters_lopmr_spd_spai.npz")

iters = len(pcg_iters_pmr)

fig, ax = plt.subplots(figsize=(5.5, 3))

ax.xaxis.set_major_locator(MaxNLocator(integer=True))

ax.plot(np.arange(0, iters), pcg_iters_pmr, color='k', linestyle=(0, (3, 1, 1, 1, 1, 1)), label='PCG (PMR_SPD SPAI)', linewidth=2)
ax.plot(np.arange(0, iters), pcg_iters_pcg, 'k:', label='PCG (PCG SPAI)', linewidth=2)
ax.plot(np.arange(0, iters), pcg_iters_lopmr, 'k-', label='PCG (LOPMR_SPD SPAI)', linewidth=2)

ax.set_xlabel('Global SPAI iteration', fontsize=fontsize)
ax.set_ylabel('PCG iterations', fontsize=fontsize)
#ax.set_title(matrix, fontsize=fontsize)
ax.legend(fontsize=fontsize)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=fontsize)

plt.tight_layout()
plt.savefig("img/Experiment07_" + matrix + "_pcg-iters.png", bbox_inches='tight', dpi=300)