from typing import final
import numpy as np
from generate_data import generate_data_sirhd
from run_model import run_model
import argparse
import os
import matplotlib.pyplot as plt
import time as time


def chi_sq(data, model, std_s):
    # Reduced chi-squared of model given some data
    chi_sq = []
    for i in range(len(data)):
        if std_s[i] != 0:
            chi_sq.append((data[i] - model[i])**2 / std_s[i]**2)
    
    return (1/len(chi_sq))*np.sum(chi_sq)


def generated_data_sirhd(model_params, state_params, added_noise_f=0.05):
    '''
        model_params = (b, g, h, d, stochastic_noise_f)
        state_params = (population, infected, nat_imm_rate, days) 
    '''
    model_params_no_noise = list(model_params).copy()
    model_params_no_noise[-1] = 0
    # Generated data with no noise
    generated_model_no_noise = generate_data_sirhd(model_params_no_noise, state_params)[0]

    # Generated data with same params and hyperparams as above, however, with noise (stochastic + added)
    generated_model_noise = generate_data_sirhd(model_params, state_params)[0]
    generated_model_noise = np.random.normal(generated_model_noise, added_noise_f*generated_model_noise)
    stds = (added_noise_f+model_params[-1])*generated_model_no_noise  # Pseudo-backprogpagation of stds (from noise)
    weights = np.array([np.std(compartment) for compartment in generated_model_noise])
    weights = (1/np.sum(weights))*weights
    
    return generated_model_no_noise, generated_model_noise, stds, weights


def mcmc(model_params, model_hyperparams, generated_data, temperature=1, iterations=20000, chains=6, \
    proposal_width_fraction=0.2, steps_to_update=100, depth_cov_mat=50, convergence_criterion=1.05):
    
    initial_params = np.full((chains, len(model_params[:-1])), model_params[:-1])  # instantiate set of parameters per chain
    old_params = np.random.uniform(0.5*initial_params, 2*initial_params)  # initialise set of parameters; following a uniform distribution
    first_params = old_params.copy()
    old_results = np.array([run_model('sirhd', params, model_hyperparams) for params in old_params])
    S_old, I_old, R_old, H_old, D_old = old_results[:,0], old_results[:,1], old_results[:,2], old_results[:,3], old_results[:,4]
    data, std, w = generated_data
    
    old_chi = [w[0]*chi_sq(data[0], s, std[0]) + w[1]*chi_sq(data[1], i, std[1]) + w[2]*chi_sq(data[2], r, std[2]) + \
        w[3]*chi_sq(data[3], h, std[3]) + w[4]*chi_sq(data[4], d, std[4]) for s, i, r, h, d in zip(S_old, I_old, R_old, H_old, D_old)]
    
    accepted_params = [[[o_p, o_c]] for o_p, o_c in zip(old_params, old_chi)]

    min_acc_steps = np.zeros((chains,))
    eigenvecs = None
    gelman_rubin = []
    for i in range(iterations):
        if (1+min(min_acc_steps))%steps_to_update == 0:
            params = []  # An array to hold vectors of values for each parameter from all chains
            for j in range(first_params.shape[1]):  # Looping through all parameters
                tmp = []
                for c in range(chains):  # Looping through all chains to stack values for each parameter from all chains
                    arr = []
                    for i in range(-int((1+min(min_acc_steps))/steps_to_update)*depth_cov_mat, 0):
                        arr.append(accepted_params[c][i][0][j])
                    tmp.append(np.array(arr) - np.mean(arr))
                    # tmp.append([accepted_params[c][i][0][j] for i in range(-int((1+min(min_acc_steps))/steps_to_update)*depth_cov_mat, 0)])
                tmp = np.array(tmp)
                params.append(tmp.flatten())
            params = np.array(params)
            
            # Obtaining the covariance matrix (bias means we use population variance equation, divide by N)
            covariance_matrix = np.cov(params, bias=True)
            eigenvals, eigenvecs = np.linalg.eig(covariance_matrix)  # Obtaining the eigenvalues and eigenvectors of the covariance matrix
            
        if not isinstance(eigenvecs, np.ndarray):
            new_params = abs(np.random.normal(old_params, proposal_width_fraction*initial_params))  # next guess is Gaussian centred at old_params
        else:
            # Taking random steps along the eigenvectors, scaled by the corresponding eigenvalues for each eigenvector
            steps = np.random.normal(0, np.sqrt(eigenvals)).reshape(-1, 1)
            param_steps = eigenvecs.T @ steps
            param_steps = param_steps.reshape(param_steps.shape[0],)
            print(param_steps)
            new_params = list(abs(old_params + param_steps))
            print(new_params)

        new_results = np.array([run_model('sirhd', params, model_hyperparams) for params in new_params])
        S_new, I_new, R_new, H_new, D_new = new_results[:,0], new_results[:,1], new_results[:,2], new_results[:,3], new_results[:,4]
        
        new_chi = [w[0]*chi_sq(data[0], s, std[0]) + w[1]*chi_sq(data[1], i, std[1]) + w[2]*chi_sq(data[2], r, std[2]) + \
        w[3]*chi_sq(data[3], h, std[3]) + w[4]*chi_sq(data[4], d, std[4]) for s, i, r, h, d in zip(S_new, I_new, R_new, H_new, D_new)]
        new_chi += np.array([5*0.5*((new_params[c][2] - 0.003428)/0.0007)**2 for c in range(chains)])  # Prior in 3rd parameter \eta
        new_chi += np.array([5*0.5*((new_params[c][3] - 0.025)/0.005)**2 for c in range(chains)])  # Prior in 4th parameter \delta

        for chain in range(chains):
            if new_chi[chain] < old_chi[chain]:
                accepted_params[chain].append([new_params[chain], new_chi[chain]])
                old_params[chain] = new_params[chain]
                old_chi[chain] = new_chi[chain]
                min_acc_steps[chain] += 1
            else:
                if np.random.binomial(1, np.exp((old_chi[chain]-new_chi[chain])/temperature)) == 1:
                    accepted_params[chain].append([new_params[chain], new_chi[chain]])
                    old_params[chain] = new_params[chain]
                    old_chi[chain] = new_chi[chain]
                    min_acc_steps[chain] += 1
                else:
                    pass
        
        # Implementing the Gelman-Rubin statistic
        N = np.mean(min_acc_steps)
        p1_data = [[accepted_params[c][i][0][0] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p1_means = [np.mean(el) for el in p1_data]
        p1_stds = [np.std(el) for el in p1_data]
        p1_mean = np.mean(p1_means)
        B_1 = (N / (chains - 1)) * np.sum([[(p1_m - p1_mean)**2] for p1_m in p1_means])
        W_1 = (1/chains) * np.sum([p1_s**2 for p1_s in p1_stds])
        V_1 = ((N-1)/N)*W_1 + ((chains+1)/(chains*N))*B_1
        
        p2_data = [[accepted_params[c][i][0][1] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p2_means = [np.mean(el) for el in p2_data]
        p2_stds = [np.std(el) for el in p2_data]
        p2_mean = np.mean(p2_means)
        B_2 = (N / (chains - 1)) * np.sum([[(p2_m - p2_mean)**2] for p2_m in p2_means])
        W_2 = (1/chains) * np.sum([p2_s**2 for p2_s in p2_stds])
        V_2 = ((N-1)/N)*W_2 + ((chains+1)/(chains*N))*B_2
        
        p3_data = [[accepted_params[c][i][0][2] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p3_means = [np.mean(el) for el in p3_data]
        p3_stds = [np.std(el) for el in p3_data]
        p3_mean = np.mean(p3_means)
        B_3 = (N / (chains - 1)) * np.sum([[(p3_m - p3_mean)**2] for p3_m in p3_means])
        W_3 = (1/chains) * np.sum([p3_s**2 for p3_s in p3_stds])
        V_3 = ((N-1)/N)*W_3 + ((chains+1)/(chains*N))*B_3
        
        p4_data = [[accepted_params[c][i][0][3] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p4_means = [np.mean(el) for el in p4_data]
        p4_stds = [np.std(el) for el in p4_data]
        p4_mean = np.mean(p4_means)
        B_4 = (N / (chains - 1)) * np.sum([[(p4_m - p4_mean)**2] for p4_m in p4_means])
        W_4 = (1/chains) * np.sum([p4_s**2 for p4_s in p4_stds])
        V_4 = ((N-1)/N)*W_4 + ((chains+1)/(chains*N))*B_4
        
        gelman_rubin.append([V_1/W_1, V_2/W_2, V_3/W_3, V_4/W_4])
        if (np.array(gelman_rubin[-1]) < convergence_criterion).all():
            break
        print('Gelman-Rubin: ', gelman_rubin[-1])
        print('Done:', i+1, '/' + str(iterations), '  ||  Accepted:', min_acc_steps)
    
    for chain in range(chains):
        with open('Chains/chain_'+str(chain+1)+'.txt', 'w') as file:
            for i, step in enumerate(accepted_params[chain]):
                file.write(str(step[1])+',')
                for j, param in enumerate(step[0]):
                    file.write(str(param))
                    if j != len(step[0])-1:
                        file.write(',')
                file.write('\n')

    return accepted_params, first_params, gelman_rubin


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: MCMC Algorithm 3',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-w', '--proposal_width_fraction', type=float, help='Standard deviation of search, as a fraction of value')
    parser.add_argument('-c', '--chains', type=int, help='Number of chains for search')
    parser.add_argument('-i', '--iterations', type=int, help='Number of iterations for search')
    parser.add_argument('-t', '--temperature', type=float, help='Temperature of Boltzmann term')
    parser.add_argument('-su', '--steps_to_update', type=int, help='how many steps before re-updating covariance matrix')
    parser.add_argument('-dc', '--depth_cov', type=int, help='Depth for covariance matrix - number of accepted points per chain')
    parser.add_argument('-cc', '--convergence_criterion', type=float, help='Convergence criterion for Gelman-Rubin statistic')
    args = parser.parse_args()  # Parses all arguments provided at script on command-line
    
    if args.proposal_width_fraction == None:
        args.proposal_width_fraction = 0.2
    if args.chains == None:
        args.chains = 8
    if args.iterations == None:
        args.iterations = 10000
    if args.temperature == None:
        args.temperature = 1
    if args.steps_to_update == None:
        args.steps_to_update = 100
    if args.depth_cov == None:
        args.depth_cov = 50
    if args.convergence_criterion == None:
        args.convergence_criterion = 1.1
    
    with open('model_params.txt', 'r') as file:
        lines = file.readlines()
    model_params = [float(lines[i].rstrip('\n').split(' ')[-1]) for i in range(len(lines))]

    with open('hyperparams.txt', 'r') as file:
        lines = file.readlines()
    hyperparams = [float(lines[i].rstrip('\n').split(' ')[-1]) for i in range(len(lines))]
    hyperparams[-1] = int(hyperparams[-1])

    generated_data_no_noise, generated_data_noise, generated_data_stds, generated_data_weights = generated_data_sirhd(model_params, hyperparams)
    generated_data = [generated_data_noise, generated_data_stds, generated_data_weights]
    s = time.time()
    accepted_params, first_params, gelman_rubin = mcmc(model_params, hyperparams, generated_data, temperature=args.temperature, \
        iterations=args.iterations, chains=args.chains, proposal_width_fraction=args.proposal_width_fraction, \
            steps_to_update=args.steps_to_update, depth_cov_mat=args.depth_cov, convergence_criterion=args.convergence_criterion)
    e = time.time()
    print('Time taken: ', round(e-s, 3))
    number_of_acc_steps = [len(a) for a in accepted_params]
    proportion_accepted = [round(n/args.iterations*100, 2) for n in number_of_acc_steps]
    print(str('%') + ' of points accepted: ', proportion_accepted)

    np.save(os.path.join('ExperimentData', 'model_params'), np.array(model_params))
    np.save(os.path.join('ExperimentData', 'hyperparams'), np.array(hyperparams))
    np.save(os.path.join('ExperimentData', 'generated_data_no_noise'), np.array(generated_data_no_noise))
    np.save(os.path.join('ExperimentData', 'generated_data_noise'), np.array(generated_data_noise))
    np.save(os.path.join('ExperimentData', 'generated_data_stds'), np.array(generated_data_stds))
    np.save(os.path.join('ExperimentData', 'accepted_params'), np.array(accepted_params))
    np.save(os.path.join('ExperimentData', 'first_params'), np.array(first_params))
    np.save(os.path.join('ExperimentData', 'number_of_acc_steps'), np.array(number_of_acc_steps))
    np.save(os.path.join('ExperimentData', 'gelman_rubin'), np.array(gelman_rubin))
