from typing import final
import numpy as np
from data import get_real_data
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


def mcmc(model_params, model_hyperparams, real_infected, temperature=1, iterations=20000, chains=6, \
    proposal_width_fraction=0.2, steps_to_update=100, depth_cov_mat=50, convergence_criterion=1.05):
    
    initial_params = np.full((chains, len(model_params[:-1])), model_params[:-1])  # instantiate set of parameters per chain
    old_params = np.random.uniform(0.95*initial_params, 1.05*initial_params)  # initialise set of parameters; following a uniform distribution
    first_params = old_params.copy()
    old_results = np.array([run_model('sirhd', params, model_hyperparams) for params in old_params])
    I_old = old_results[:,1]

    stds = 0.02*np.array(real_infected)
    old_chi = [chi_sq(real_infected, i, stds) for i in I_old]

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
        I_new = new_results[:,1]
        
        new_chi = [chi_sq(real_infected, i, stds) for i in I_new]
        new_chi += np.array([0.5*((new_params[c][18] - 0.16667)/0.016667)**2 for c in range(chains)])
        new_chi += np.array([0.5*((new_params[c][19] - 0.003428)/0.0003428)**2 for c in range(chains)])
        new_chi += np.array([0.5*((new_params[c][20] - 0.025)/0.005)**2 for c in range(chains)])

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

        p5_data = [[accepted_params[c][i][0][4] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p5_means = [np.mean(el) for el in p5_data]
        p5_stds = [np.std(el) for el in p5_data]
        p5_mean = np.mean(p5_means)
        B_5 = (N / (chains - 1)) * np.sum([[(p5_m - p5_mean)**2] for p5_m in p5_means])
        W_5 = (1/chains) * np.sum([p5_s**2 for p5_s in p5_stds])
        V_5 = ((N-1)/N)*W_5 + ((chains+1)/(chains*N))*B_5

        p6_data = [[accepted_params[c][i][0][5] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p6_means = [np.mean(el) for el in p6_data]
        p6_stds = [np.std(el) for el in p6_data]
        p6_mean = np.mean(p6_means)
        B_6 = (N / (chains - 1)) * np.sum([[(p6_m - p6_mean)**2] for p6_m in p6_means])
        W_6 = (1/chains) * np.sum([p6_s**2 for p6_s in p6_stds])
        V_6 = ((N-1)/N)*W_6 + ((chains+1)/(chains*N))*B_6

        p7_data = [[accepted_params[c][i][0][6] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p7_means = [np.mean(el) for el in p7_data]
        p7_stds = [np.std(el) for el in p7_data]
        p7_mean = np.mean(p7_means)
        B_7 = (N / (chains - 1)) * np.sum([[(p7_m - p7_mean)**2] for p7_m in p7_means])
        W_7 = (1/chains) * np.sum([p7_s**2 for p7_s in p7_stds])
        V_7 = ((N-1)/N)*W_7 + ((chains+1)/(chains*N))*B_7

        p8_data = [[accepted_params[c][i][0][7] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p8_means = [np.mean(el) for el in p8_data]
        p8_stds = [np.std(el) for el in p8_data]
        p8_mean = np.mean(p8_means)
        B_8 = (N / (chains - 1)) * np.sum([[(p8_m - p8_mean)**2] for p8_m in p8_means])
        W_8 = (1/chains) * np.sum([p8_s**2 for p8_s in p8_stds])
        V_8 = ((N-1)/N)*W_8 + ((chains+1)/(chains*N))*B_8

        p9_data = [[accepted_params[c][i][0][8] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p9_means = [np.mean(el) for el in p9_data]
        p9_stds = [np.std(el) for el in p9_data]
        p9_mean = np.mean(p9_means)
        B_9 = (N / (chains - 1)) * np.sum([[(p9_m - p9_mean)**2] for p9_m in p9_means])
        W_9 = (1/chains) * np.sum([p9_s**2 for p9_s in p9_stds])
        V_9 = ((N-1)/N)*W_9 + ((chains+1)/(chains*N))*B_9

        p10_data = [[accepted_params[c][i][0][9] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p10_means = [np.mean(el) for el in p10_data]
        p10_stds = [np.std(el) for el in p10_data]
        p10_mean = np.mean(p10_means)
        B_10 = (N / (chains - 1)) * np.sum([[(p10_m - p10_mean)**2] for p10_m in p10_means])
        W_10 = (1/chains) * np.sum([p10_s**2 for p10_s in p10_stds])
        V_10 = ((N-1)/N)*W_10 + ((chains+1)/(chains*N))*B_10

        p11_data = [[accepted_params[c][i][0][10] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p11_means = [np.mean(el) for el in p11_data]
        p11_stds = [np.std(el) for el in p11_data]
        p11_mean = np.mean(p11_means)
        B_11 = (N / (chains - 1)) * np.sum([[(p11_m - p11_mean)**2] for p11_m in p11_means])
        W_11 = (1/chains) * np.sum([p11_s**2 for p11_s in p11_stds])
        V_11 = ((N-1)/N)*W_11 + ((chains+1)/(chains*N))*B_11

        p12_data = [[accepted_params[c][i][0][11] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p12_means = [np.mean(el) for el in p12_data]
        p12_stds = [np.std(el) for el in p12_data]
        p12_mean = np.mean(p12_means)
        B_12 = (N / (chains - 1)) * np.sum([[(p12_m - p12_mean)**2] for p12_m in p12_means])
        W_12 = (1/chains) * np.sum([p12_s**2 for p12_s in p12_stds])
        V_12 = ((N-1)/N)*W_12 + ((chains+1)/(chains*N))*B_12

        p13_data = [[accepted_params[c][i][0][12] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p13_means = [np.mean(el) for el in p13_data]
        p13_stds = [np.std(el) for el in p13_data]
        p13_mean = np.mean(p13_means)
        B_13 = (N / (chains - 1)) * np.sum([[(p13_m - p13_mean)**2] for p13_m in p13_means])
        W_13 = (1/chains) * np.sum([p13_s**2 for p13_s in p13_stds])
        V_13 = ((N-1)/N)*W_13 + ((chains+1)/(chains*N))*B_13

        p14_data = [[accepted_params[c][i][0][13] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p14_means = [np.mean(el) for el in p14_data]
        p14_stds = [np.std(el) for el in p14_data]
        p14_mean = np.mean(p14_means)
        B_14 = (N / (chains - 1)) * np.sum([[(p14_m - p14_mean)**2] for p14_m in p14_means])
        W_14 = (1/chains) * np.sum([p14_s**2 for p14_s in p14_stds])
        V_14 = ((N-1)/N)*W_14 + ((chains+1)/(chains*N))*B_14

        p15_data = [[accepted_params[c][i][0][14] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p15_means = [np.mean(el) for el in p15_data]
        p15_stds = [np.std(el) for el in p15_data]
        p15_mean = np.mean(p15_means)
        B_15 = (N / (chains - 1)) * np.sum([[(p15_m - p15_mean)**2] for p15_m in p15_means])
        W_15 = (1/chains) * np.sum([p15_s**2 for p15_s in p15_stds])
        V_15 = ((N-1)/N)*W_15 + ((chains+1)/(chains*N))*B_15

        p16_data = [[accepted_params[c][i][0][15] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p16_means = [np.mean(el) for el in p16_data]
        p16_stds = [np.std(el) for el in p16_data]
        p16_mean = np.mean(p16_means)
        B_16 = (N / (chains - 1)) * np.sum([[(p16_m - p16_mean)**2] for p16_m in p16_means])
        W_16 = (1/chains) * np.sum([p16_s**2 for p16_s in p16_stds])
        V_16 = ((N-1)/N)*W_16 + ((chains+1)/(chains*N))*B_16

        p17_data = [[accepted_params[c][i][0][16] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p17_means = [np.mean(el) for el in p17_data]
        p17_stds = [np.std(el) for el in p17_data]
        p17_mean = np.mean(p17_means)
        B_17 = (N / (chains - 1)) * np.sum([[(p17_m - p17_mean)**2] for p17_m in p17_means])
        W_17 = (1/chains) * np.sum([p17_s**2 for p17_s in p17_stds])
        V_17 = ((N-1)/N)*W_17 + ((chains+1)/(chains*N))*B_17

        p18_data = [[accepted_params[c][i][0][17] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p18_means = [np.mean(el) for el in p18_data]
        p18_stds = [np.std(el) for el in p18_data]
        p18_mean = np.mean(p18_means)
        B_18 = (N / (chains - 1)) * np.sum([[(p18_m - p18_mean)**2] for p18_m in p18_means])
        W_18 = (1/chains) * np.sum([p18_s**2 for p18_s in p18_stds])
        V_18 = ((N-1)/N)*W_18 + ((chains+1)/(chains*N))*B_18

        p19_data = [[accepted_params[c][i][0][18] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p19_means = [np.mean(el) for el in p19_data]
        p19_stds = [np.std(el) for el in p19_data]
        p19_mean = np.mean(p19_means)
        B_19 = (N / (chains - 1)) * np.sum([[(p19_m - p19_mean)**2] for p19_m in p19_means])
        W_19 = (1/chains) * np.sum([p19_s**2 for p19_s in p19_stds])
        V_19 = ((N-1)/N)*W_19 + ((chains+1)/(chains*N))*B_19

        p20_data = [[accepted_params[c][i][0][19] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p20_means = [np.mean(el) for el in p20_data]
        p20_stds = [np.std(el) for el in p20_data]
        p20_mean = np.mean(p20_means)
        B_20 = (N / (chains - 1)) * np.sum([[(p20_m - p20_mean)**2] for p20_m in p20_means])
        W_20 = (1/chains) * np.sum([p20_s**2 for p20_s in p20_stds])
        V_20 = ((N-1)/N)*W_20 + ((chains+1)/(chains*N))*B_20

        p21_data = [[accepted_params[c][i][0][20] for i in range(len(accepted_params[c]))] for c in range(chains)]
        p21_means = [np.mean(el) for el in p21_data]
        p21_stds = [np.std(el) for el in p21_data]
        p21_mean = np.mean(p21_means)
        B_21 = (N / (chains - 1)) * np.sum([[(p21_m - p21_mean)**2] for p21_m in p21_means])
        W_21 = (1/chains) * np.sum([p21_s**2 for p21_s in p21_stds])
        V_21 = ((N-1)/N)*W_21 + ((chains+1)/(chains*N))*B_21
        
        # gelman_rubin.append([V_1/W_1, V_2/W_2, V_3/W_3, V_4/W_4, V_5/W_5, V_6/W_6, V_7/W_7, V_8/W_8, V_9/W_9, V_10/W_10, \
            # V_11/W_11, V_12/W_12, V_13/W_13])
        gelman_rubin.append([V_1/W_1, V_2/W_2, V_3/W_3, V_4/W_4, V_5/W_5, V_6/W_6, V_7/W_7, V_8/W_8, V_9/W_9, V_10/W_10, V_11/W_11, V_12/W_12, V_13/W_13, V_14/W_14, V_15/W_15, V_16/W_16, V_17/W_17, V_18/W_18, V_19/W_19, V_20/W_20, V_21/W_21])

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

    s = time.time()
    real_infected = get_real_data(hyperparams[0])
    
    accepted_params, first_params, gelman_rubin = mcmc(model_params, hyperparams, real_infected, temperature=args.temperature, \
        iterations=args.iterations, chains=args.chains, proposal_width_fraction=args.proposal_width_fraction, \
            steps_to_update=args.steps_to_update, depth_cov_mat=args.depth_cov, convergence_criterion=args.convergence_criterion)
    
    e = time.time()
    print('Time taken: ', round(e-s, 3))
    
    number_of_acc_steps = [len(a) for a in accepted_params]
    proportion_accepted = [round(n/args.iterations*100, 2) for n in number_of_acc_steps]
    print(str('%') + ' of points accepted: ', proportion_accepted)

    np.save(os.path.join('ExperimentData', 'model_params'), np.array(model_params))
    np.save(os.path.join('ExperimentData', 'hyperparams'), np.array(hyperparams))
    np.save(os.path.join('ExperimentData', 'real_infected'), np.array(real_infected))
    np.save(os.path.join('ExperimentData', 'accepted_params'), np.array(accepted_params))
    np.save(os.path.join('ExperimentData', 'first_params'), np.array(first_params))
    np.save(os.path.join('ExperimentData', 'number_of_acc_steps'), np.array(number_of_acc_steps))
    np.save(os.path.join('ExperimentData', 'gelman_rubin'), np.array(gelman_rubin))
