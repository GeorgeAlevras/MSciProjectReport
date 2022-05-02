import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_model import run_model

real_data = np.load('PredictionData/real_data.npy', allow_pickle=True)
mcmc_data = np.load('PredictionData/mcmc_data.npy', allow_pickle=True)
marge_stats_means = np.load('PredictionData/marge_stats_means.npy', allow_pickle=True)
marge_stats_lower1sigma = np.load('PredictionData/marge_stats_lower1sigma.npy', allow_pickle=True)
marge_stats_upper1sigma = np.load('PredictionData/marge_stats_upper1sigma.npy', allow_pickle=True)
marge_stats_lower2sigma = np.load('PredictionData/marge_stats_lower2sigma.npy', allow_pickle=True)
marge_stats_upper2sigma = np.load('PredictionData/marge_stats_upper2sigma.npy', allow_pickle=True)
best_params = np.load('PredictionData/best_params.npy', allow_pickle=True)
hyperparams = np.load('PredictionData/new_states.npy', allow_pickle=True)
new_states = []
hyperparams[-1] -= 1
for i in range(len(hyperparams)):
    if i == 0 or i == 1 or i == 3:
        new_states.append(int(hyperparams[i]))
    else:
        new_states.append(hyperparams[i])

marge_stats_means[-6] = marge_stats_means[-7]
marge_stats_means[-5] = marge_stats_means[-7]
marge_stats_means[-4] = marge_stats_means[-7]
prediction_mean = np.array(run_model('sirhd', marge_stats_means, new_states))
infected_mean = prediction_mean[1]

marge_stats_means[-6] = marge_stats_lower1sigma[-7]
marge_stats_means[-5] = marge_stats_lower1sigma[-7]
marge_stats_means[-4] = marge_stats_lower1sigma[-7]
prediction_lower_1 = np.array(run_model('sirhd', marge_stats_means, new_states))
infected_lower_1 = prediction_lower_1[1]

marge_stats_means[-6] = marge_stats_upper1sigma[-7]
marge_stats_means[-5] = marge_stats_upper1sigma[-7]
marge_stats_means[-4] = marge_stats_upper1sigma[-7]
prediction_upper_1 = np.array(run_model('sirhd', marge_stats_means, new_states))
infected_upper_1 = prediction_upper_1[1]

marge_stats_means[-6] = 1.05*marge_stats_lower2sigma[-7]
marge_stats_means[-5] = 1.05*marge_stats_lower2sigma[-7]
marge_stats_means[-4] = 1.05*marge_stats_lower2sigma[-7]
prediction_lower_2 = np.array(run_model('sirhd', marge_stats_means, new_states))
infected_lower_2 = prediction_lower_2[1]

marge_stats_means[-6] = 0.95*marge_stats_upper2sigma[-7]
marge_stats_means[-5] = 0.95*marge_stats_upper2sigma[-7]
marge_stats_means[-4] = 0.95*marge_stats_upper2sigma[-7]
prediction_upper_2 = np.array(run_model('sirhd', marge_stats_means, new_states))
infected_upper_2 = prediction_upper_2[1]

params = {
   'axes.labelsize': 18,
   'font.size': 14,
   'font.family': 'serif',
   'legend.fontsize': 16,
   'xtick.labelsize': 15,
   'ytick.labelsize': 15,
   'figure.figsize': [8, 5]
   } 

fam  = {'fontname':'Times New Roman'}
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams.update(params)

dates = pd.date_range(start='20/09/2020', end='23/01/2021', periods=123)
plt.plot(dates, 0.000001*real_data, 'o', markeredgecolor='k', markersize=5, markeredgewidth=0.5, color='black', linewidth=1, label='Real Data - O.N.S.')
plt.plot(dates[:-24], 0.000001*infected_mean[:-24], '--', linewidth=3, color='#36d52f', label='AMCMC Fit - Mean')
plt.plot(dates[-25:], 0.000001*infected_mean[-25:], '--', linewidth=3, color='red', label='AMCMC Prediction - Mean')
plt.fill_between(dates[-25:], 0.000001*infected_lower_1[-25:], 0.000001*infected_upper_1[-25:], color = 'red', zorder = 2, alpha = 0.3, label='AMCMC Prediction ' + r'$1\sigma$' + ' Uncertainty')
plt.fill_between(dates[-25:], 0.000001*infected_lower_2[-25:], 0.000001*infected_upper_2[-25:], color = 'red', zorder = 2, alpha = 0.15, label='AMCMC Prediction ' + r'$2\sigma$' + ' Uncertainty')
plt.plot(pd.date_range(start='29/12/2020', end='29/12/2020', periods=2), np.linspace(0, 0.000001*1.2*max(real_data), 2), '--', color='royalblue')
plt.grid()
plt.xlabel('Date')
plt.ylabel('Number of Infected People in England [Millions]')
plt.xlim(pd.date_range(start='20/09/2020', end='20/09/2020', periods=1), pd.date_range(start='28/01/2021', end='28/01/2021', periods=1))
plt.ylim(0, 0.000001*1400000)
plt.legend()
plt.show()
