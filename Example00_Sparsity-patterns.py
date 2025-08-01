import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread
import matplotlib.gridspec as gridspec

plt.rc('text.latex', preamble=r'\usepackage{amssymb} \usepackage{amsmath}')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fontsize = 10

matrix_source = "../matrix-market/"

matrices = ("bcsstk21", "Poisson4k", "bcsstk15", "Poisson8k", "Poisson16k",
            "bodyy6", "Poisson32k", "Poisson64k", "Poisson128k", "tmt_sym")

mtx_files = ("bcsstk21.mtx",
             "Poisson_SExp_sig21.0_L0.1_DoF4000_K.mtx",
             "bcsstk15.mtx",
             "Poisson_SExp_sig21.0_L0.1_DoF8000_K.mtx",
             "Poisson_SExp_sig21.0_L0.1_DoF16000_K.mtx",
             "bodyy6.mtx",
             "Poisson_SExp_sig21.0_L0.1_DoF32000_K.mtx",
             "Poisson_SExp_sig21.0_L0.1_DoF64000_K.mtx",
             "Poisson_SExp_sig21.0_L0.1_DoF128000_K.mtx",
             "tmt_sym.mtx")

fig = plt.figure(figsize=(6, 3))
gs = gridspec.GridSpec(2, 5, wspace=0.1, hspace=0.03)

for i, (fname, mtx) in enumerate(zip(mtx_files, matrices)):
    row = i // 5 
    col = i % 5
    
    ax = fig.add_subplot(gs[row, col])
    
    M = mmread(matrix_source +  fname)

    if False:
      ax.spy(M, markersize=0.05, color='black', marker='.')
    else:
      coo = M.tocoo()
      ax.scatter(coo.col, coo.row, s=.1, c='black', marker='o', 
                 edgecolors='none', linewidths=0)
      ax.invert_yaxis()
      ax.set_aspect('equal')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(mtx, fontsize=fontsize)

plt.tight_layout()
plt.savefig("img/sparsity-patterns.png", bbox_inches='tight', dpi=300)