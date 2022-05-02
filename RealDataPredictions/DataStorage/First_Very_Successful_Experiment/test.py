import numpy as np
from data import get_real_data
from run_model import run_model
import matplotlib.pyplot as plt


with open('model_params.txt', 'r') as file:
        lines = file.readlines()
model_params = [float(lines[i].rstrip('\n').split(' ')[-1]) for i in range(len(lines))]

with open('hyperparams.txt', 'r') as file:
        lines = file.readlines()
hyperparams = [float(lines[i].rstrip('\n').split(' ')[-1]) for i in range(len(lines))]
hyperparams[-1] = int(hyperparams[-1])


real_data = get_real_data(population=56000000)
print(len(real_data))
model_params = [0.26, 0.25, 0.24, 0.23, 0.23, 0.21, 0.22, 0.205, 0.195, 0.185, 0.25, 0.25, 0.28, 0.3, 0.23, 0.24, 0.25, 0.24, 0.16667, 0.003428, 0.025, 0.02]
S, I, H, R, D = run_model('sirhd', model_params[:-1], hyperparams)

plt.plot(I, label='MCMC')
plt.plot(real_data, label='Real Data')
plt.legend()
plt.show()
