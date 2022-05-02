from run_model import run_model
import numpy as np
import time as time

model_hyperparams = [1000000, 1000, 0, 100]

new_params = np.array([
    [224, 1673, 0.044, 0.08],
    [224, 1673, 0.044, 0.08],
    [224, 1673, 0.044, 0.08],
    [224, 1673, 0.044, 0.08]
])

for i in range(10000):
    s = time.time()
    new_results = np.array([run_model('sirhd', params, model_hyperparams) for params in new_params])
    e = time.time()
    print('Done: ', i, ' time: ', e-s)
