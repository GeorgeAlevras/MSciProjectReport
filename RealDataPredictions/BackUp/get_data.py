from run_model import run_model
import numpy as np


def get_data_sirhd(model_params, model_hyperparams):
    '''
        model_params = (b1, b2, b3, b4, b5, b6, b7, b8, g, h, d, stochastic_noise_f)
        model_hyperparams = [population, infected, nat_imm_rate, days]
    '''
    # Generate noisy data for simulation of real data
    generated_model = np.array(run_model('sirhd_noise', model_params, model_hyperparams))
    S, I, R, H, D = generated_model
    N = S + I + R + H

    return generated_model, N
