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

marge_stats_means[-6] = marge_stats_means[-7]  # beta17 = beta18
marge_stats_means[-5] = marge_stats_means[-7]  # beta16 = beta18
marge_stats_means[-4] = marge_stats_means[-7]  #beta15 = beta18
print(marge_stats_lower1sigma)
print(marge_stats_lower2sigma)
print(marge_stats_upper1sigma)
print(marge_stats_upper2sigma)
prediction_mean = np.array(run_model('sirhd', marge_stats_means, new_states))
infected_mean = prediction_mean[1]


beta_min = marge_stats_lower2sigma[-7]
beta_max = marge_stats_upper2sigma[-7]
gamma_min = marge_stats_lower2sigma[-3]
gamma_max = marge_stats_upper2sigma[-3]
eta_min = marge_stats_lower2sigma[-2]
eta_max = marge_stats_upper2sigma[-2]
delta_min = marge_stats_lower2sigma[-1]
delta_max = marge_stats_upper2sigma[-1]

monte_carlo_infections = []
for i in range(400):
    beta = np.random.uniform(beta_min, beta_max)
    # gamma = np.random.uniform(gamma_min, gamma_max)
    # eta = np.random.uniform(eta_min, eta_max)
    # delta = np.random.uniform(delta_min, delta_max)
    marge_stats_means[-6] = beta
    marge_stats_means[-5] = beta
    marge_stats_means[-4] = beta
    # marge_stats_means[-3] = gamma
    # marge_stats_means[-2] = eta
    # marge_stats_means[-1] = delta
    prediction_x = np.array(run_model('sirhd', marge_stats_means, new_states))
    monte_carlo_infections.append(prediction_x[1])



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
   'legend.fontsize': 19,
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
plt.plot(dates, 0.000001*real_data, 'o', markeredgecolor='k', markersize=5, markeredgewidth=0.5, color='black', linewidth=1, label='Real Data from O.N.S. - '+r'$I(t)$')
plt.plot(dates[:-24], 0.000001*infected_mean[:-24], '--', linewidth=3, color='#36d52f', label='Model Fit Using Marginal Means - '+r'$I(t)$')
for i in range(len(monte_carlo_infections)):  #[-25:]
    if i == 1:
        plt.plot(dates[-25:], 0.000001*monte_carlo_infections[i][-25:], linewidth=1, alpha=0.3, color='orange', label='Monte Carlo Simulations Across '+r'$2\sigma$'+' Range of Parameters - '+r'$I(t)$')
    plt.plot(dates[-25:], 0.000001*monte_carlo_infections[i][-25:], linewidth=1, alpha=0.3, color='orange')
plt.plot(dates[-25:], 0.000001*infected_mean[-25:], '--', linewidth=3, color='red', label='Model Prediction Using Marginal Means of Last Week - '+r'$I(t)$')
# plt.fill_between(dates[-25:], 0.000001*infected_lower_1[-25:], 0.000001*infected_upper_1[-25:], color = 'red', zorder = 2, alpha = 0.3, label='Model Prediction with ' + r'$1\sigma$' + ' Uncertainty Range - '+r'$I(t)$')
# plt.fill_between(dates[-25:], 0.000001*infected_lower_2[-25:], 0.000001*infected_upper_2[-25:], color = 'red', zorder = 2, alpha = 0.15, label='Model Prediction with ' + r'$2\sigma$' + ' Uncertainty Range - '+r'$I(t)$')
plt.plot(pd.date_range(start='29/12/2020', end='29/12/2020', periods=2), np.linspace(0, 0.000001*1.2*max(real_data), 2), '--', color='royalblue', label='Beginning of Short-Term Prediction: 29/12/2020')
plt.grid()
plt.xlabel('Date', fontname='Times New Roman', fontsize=26)
plt.ylabel('Number of Infected People in England [Millions]', fontname='Times New Roman', fontsize=26)
dtrng = pd.date_range(start='20/09/2020', end='23/01/2021', periods=10)
plt.xticks(dtrng, fontsize=20, fontname='Times New Roman')
plt.yticks(fontsize=20, fontname='Times New Roman')
plt.xlim(pd.date_range(start='20/09/2020', end='20/09/2020', periods=1), pd.date_range(start='23/01/2021', end='23/01/2021', periods=1))
plt.ylim(0, 0.000001*1400000)
plt.legend()
plt.show()
