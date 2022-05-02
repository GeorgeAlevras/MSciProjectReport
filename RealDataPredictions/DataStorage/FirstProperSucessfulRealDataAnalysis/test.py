import numpy as np
from data import get_real_data
from run_model import run_model
import matplotlib.pyplot as plt
import pandas as pd


with open('model_params.txt', 'r') as file:
        lines = file.readlines()
model_params = [float(lines[i].rstrip('\n').split(' ')[-1]) for i in range(len(lines))]

with open('hyperparams.txt', 'r') as file:
        lines = file.readlines()
hyperparams = [float(lines[i].rstrip('\n').split(' ')[-1]) for i in range(len(lines))]
hyperparams[-1] = int(hyperparams[-1])


real_data = get_real_data(population=56000000)

model_params = [0.26, 0.25, 0.24, 0.23, 0.23, 0.21, 0.225, 0.205, 0.19, 0.18, 0.26, 0.245, 0.285, 0.30, 0.24, 0.24, 0.24, 0.24, 0.16667, 0.003428, 0.025, 0.02]
S, I, H, R, D = run_model('sirhd', model_params[:-1], hyperparams)

dates = pd.date_range(start='20/09/2020', end='23/01/2021', periods=123)
plt.plot(dates, I[:-1], label='MCMC')
plt.plot(dates, real_data, label='Real Data')
plt.grid()
plt.legend()
plt.show()
