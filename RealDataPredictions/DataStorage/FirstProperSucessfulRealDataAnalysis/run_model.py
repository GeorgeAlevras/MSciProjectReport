from models import models
import numpy as np
from scipy.integrate import solve_ivp


def run_model(model, model_params, model_hyperparams):
    try:
        return run_sirhd(models[model], model_params, model_hyperparams)
    except ValueError:
        print('Model not available')


def run_sirhd(odes, model_params, model_hyperparams):
    '''
        model_params = (b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, g, h, d)
        if odes have noise:
            model_params = (b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, g, h, d, stochastic_noise_f)
        model_hyperparams = [population, infected, nat_imm_rate, days]
    '''
    population, infected, nat_imm_rate, days = model_hyperparams

    t_span = np.array([0, days])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)  # Time series, unit=1 day

    # Initialisation of compartments
    susceptible = population - infected - population*nat_imm_rate
    recovered = population*nat_imm_rate
    hospitalised = 0
    deceased = 0
    x_0 = [susceptible, infected, recovered, hospitalised, deceased]
    
    # Integrator solves system of DEs (model)
    solutions = solve_ivp(odes, t_span, x_0, args=model_params, t_eval=t)
    S = solutions.y[0]
    I = solutions.y[1]
    R = solutions.y[2]
    H = solutions.y[3]
    D = solutions.y[4]

    return [S, I, R, H, D]
