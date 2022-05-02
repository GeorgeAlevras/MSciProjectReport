import matplotlib
import matplotlib.pyplot as plt
import numpy as np

gen_noise = np.load('ToPlotMyself/gen_noise.npy', allow_pickle=True)
gen_no_noise = np.load('ToPlotMyself/gen_no_noise.npy', allow_pickle=True)
infected_best = np.load('ToPlotMyself/infected_best.npy', allow_pickle=True)

fig, ax = plt.subplots()
params = {'legend.fontsize': 12}
plt.rcParams.update(params)
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'

t = np.linspace(0, len(gen_noise)-1, len(gen_noise))
plt.plot(t, gen_noise/1e6, 'o', markeredgecolor='k', markersize=6, markeredgewidth=0.5, color='#ff1313', linewidth=1, label='Noisy Simulated Data - '+r'$I(t)$')
# plt.plot(t, gen_no_noise/1e6, label='Modelled Simulated Data without Noise - '+r'$I(t)$', linewidth=3, color='#36d52f')
plt.plot(t, infected_best/1e6, '--', label='Modelled Data Using Best-Estimate Parameters ' + r'$\theta^*$'+' - '+r'$I(t)$', linewidth=2, color='black')

plt.legend(loc='upper left')
plt.xlabel('Time [days]', fontname='Times New Roman', fontsize=19)
plt.ylabel('Fraction of Population', fontname='Times New Roman', fontsize=19)
plt.grid()
plt.minorticks_on()
ax.tick_params(direction='in')
ax.tick_params(which='minor', direction='in')
plt.xticks(fontsize=17, fontname='Times New Roman')
plt.yticks(fontsize=17, fontname='Times New Roman')
plt.xlim(0, 50)
plt.ylim(0, 0.4)
plt.show()