import matplotlib.pyplot as plt
import numpy as np
plt.rc('text.latex', preamble=r'\usepackage{amssymb} \usepackage{amsmath}')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def plot_spectra_simple(spectra_dict, matrix_name, figsize=(5.5, 3.5), fontsize=10.5):
    fig, ax = plt.subplots(figsize=figsize)
    n_spectra = len(spectra_dict)
    y_positions = np.arange(n_spectra)
    
    for i, (name, spectrum) in enumerate(spectra_dict.items()):
        pos_eigs = spectrum[spectrum > 0]
        neg_eigs = spectrum[spectrum < 0]
        
        if len(pos_eigs) > 0:
            y_pos_top = np.full(len(pos_eigs), y_positions[i] + 0.35)
            y_pos_bottom = np.full(len(pos_eigs), y_positions[i] + 0.05)
            for j, eig in enumerate(pos_eigs):
                ax.plot([eig, eig], [y_pos_bottom[j], y_pos_top[j]], 'k-', linewidth=0.3, alpha=0.7)

        if len(neg_eigs) > 0:
            y_pos_top = np.full(len(neg_eigs), y_positions[i] - 0.05)
            y_pos_bottom = np.full(len(neg_eigs), y_positions[i] - 0.35)
            for j, eig in enumerate(-neg_eigs): 
                ax.plot([eig, eig], [y_pos_bottom[j], y_pos_top[j]], 'r-', linewidth=0.3, alpha=0.7)
    
    ax.set_xscale('log')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(list(spectra_dict.keys()), fontsize=fontsize)
    ax.set_xlabel('Eigenvalue magnitude', fontsize=fontsize)
    ax.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle(matrix_name, fontsize=fontsize, y=0.733, x=.55, ha='center')
    
    ax.plot([], [], 'k-', linewidth=2, label='Positive eigenvalues')
    ax.plot([], [], 'r-', linewidth=2, label='Negative eigenvalues')
    ax.legend(fontsize=fontsize-1, loc='upper center', bbox_to_anchor=(0.5, 1.27), ncol=2)

    plt.tight_layout()
    return fig


matrix = "triunif4k"  # \in {"bcsstk21", 
                      #      "tri100eigs4k", 
                      #      "triclust4k",
                      #      "triunif4k",
                      #      "msc04515"}

fontsize = 10.5

A_spectrum = np.load("data/Experiment01_" + matrix + "_spectrum.npz")**-1
M_spectrum_lopmr = np.load("data/Experiment01_" + matrix + "_spectrum_lopmr.npz")
M_spectrum_pcr = np.load("data/Experiment01_" + matrix + "_spectrum_pcr.npz")
M_spectrum_pmr = np.load("data/Experiment01_" + matrix + "_spectrum_pmr.npz")
M_spectrum_pcg = np.load("data/Experiment01_" + matrix + "_spectrum_pcg.npz")
M_spectrum_psd = np.load("data/Experiment01_" + matrix + "_spectrum_psd.npz")

spectra = {r'$A^{-1}$': A_spectrum,
           'LOPMR': M_spectrum_lopmr,
           'PCR': M_spectrum_pcr,
           'PMR': M_spectrum_pmr,
           'PCG': M_spectrum_pcg,
           'PSD': M_spectrum_psd}

fig = plot_spectra_simple(spectra, matrix, fontsize=fontsize)
plt.savefig("img/Experiment01_" + matrix + "_spectra.png", bbox_inches='tight', dpi=300)