import numpy as np
import matplotlib.pyplot as plt
import matplotlib

gelman_rubin = np.load('ExperimentData/gelman_rubin.npy', allow_pickle=True)

params = {
   'axes.labelsize': 17,
   'font.size': 14,
   'font.family': 'serif',
   'legend.fontsize': 16,
   'xtick.labelsize': 14,
   'ytick.labelsize': 14,
   'figure.figsize': [8, 5]
   } 

fam  = {'fontname':'Times New Roman'}
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams.update(params)
plt.plot(np.linspace(0, len(gelman_rubin), len(gelman_rubin)), gelman_rubin[:,0], label=r'$\beta$')
plt.plot(np.linspace(0, len(gelman_rubin), len(gelman_rubin)), gelman_rubin[:,1], label=r'$\gamma$')
plt.plot(np.linspace(0, len(gelman_rubin), len(gelman_rubin)), gelman_rubin[:,2], label=r'$\eta$')
plt.plot(np.linspace(0, len(gelman_rubin), len(gelman_rubin)), gelman_rubin[:,3], label=r'$\delta$')
plt.plot(np.linspace(1, len(gelman_rubin), len(gelman_rubin)), np.linspace(1.05, 1.05, len(gelman_rubin)), '--', linewidth=2, color='black', label=r'$R_c=1.05$')
plt.xlabel('Iteration', **fam, fontsize=17)
plt.ylabel('Gelman-Rubin Convergence Criterion', **fam, fontsize=17)
plt.grid()
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlim(1, 10000)
plt.ylim(1, 300)
plt.savefig('Images/gelman_rubin.png')

plt.show()