from __future__ import print_function
from types import new_class
import numpy as np
import sys, os
from getdist import plots, MCSamples, loadMCSamples
import getdist, IPython
import pylab as plt
import seaborn as sns
import os
from glob import glob
from getdist import plots
from run_model import run_model


if __name__ == '__main__':
    model_params = np.load('ExperimentData/model_params.npy', allow_pickle=True)
    hyperparams = np.load('ExperimentData/hyperparams.npy', allow_pickle=True)
    real_infected = np.load('ExperimentData/real_infected.npy', allow_pickle=True)
    accepted_params = np.load('ExperimentData/accepted_params.npy', allow_pickle=True)
    first_params = np.load('ExperimentData/first_params.npy', allow_pickle=True)
    number_of_acc_steps = np.load('ExperimentData/number_of_acc_steps.npy', allow_pickle=True)
    gelman_rubin = np.load('ExperimentData/gelman_rubin.npy', allow_pickle=True)
    
    sys.path.insert(0, os.path.realpath(os.path.join(os.getcwd(), '..')))
    plt.rcParams['text.usetex']=True

    sns.set()
    sns.set_style("white")
    sns.set_context("talk")
    palette = sns.color_palette()
    plt.rcParams.update({'lines.linewidth': 3})
    plt.rcParams.update({'font.size': 22})

    file_names = [y for x in os.walk('./Chains/') for y in glob(os.path.join(x[0], 'chain*'))]

    for idx, file in enumerate(file_names):
        inChain = np.loadtxt(file,delimiter=',')
        nsamps, npar = inChain.shape
        outChain = np.zeros((nsamps,npar+1))
        outChain[:,1:] = np.copy(inChain)
        outChain[:,0] = 1.
        np.savetxt('./ConvertedFiles/convert_{}.txt'.format(idx+1),outChain)

    samples = loadMCSamples('./ConvertedFiles/convert', settings={'ignore_rows':3})  # , settings={'ignore_rows':10}
    
    marge_stats = [x for x in str(samples.getMargeStats()).split(' ') if '.' in x]
    marge_stats_means = [float(marge_stats[3]), float(marge_stats[11]), float(marge_stats[19]), float(marge_stats[27]), float(marge_stats[35]), float(marge_stats[43]), float(marge_stats[51])]
    marge_stats_lower1sigma = [float(marge_stats[5]), float(marge_stats[13]), float(marge_stats[21]), float(marge_stats[29]), float(marge_stats[37]), float(marge_stats[45]), float(marge_stats[53])]
    marge_stats_upper1sigma = [float(marge_stats[6]), float(marge_stats[14]), float(marge_stats[22]), float(marge_stats[30]), float(marge_stats[38]), float(marge_stats[46]), float(marge_stats[54])]
    marge_stats_lower2sigma = [float(marge_stats[7]), float(marge_stats[15]), float(marge_stats[23]), float(marge_stats[31]), float(marge_stats[39]), float(marge_stats[47]), float(marge_stats[55])]
    marge_stats_upper2sigma = [float(marge_stats[8]), float(marge_stats[16]), float(marge_stats[24]), float(marge_stats[32]), float(marge_stats[40]), float(marge_stats[48]), float(marge_stats[56])]
    marge_stats_lower3sigma = [float(marge_stats[9]), float(marge_stats[17]), float(marge_stats[25]), float(marge_stats[33]), float(marge_stats[41]), float(marge_stats[49]), float(marge_stats[57])]
    marge_stats_upper3sigma = [float(marge_stats[10]), float(marge_stats[18]), float(marge_stats[26]), float(marge_stats[34]), float(marge_stats[42]), float(marge_stats[50]), float(marge_stats[58])]
    print('Means:', marge_stats_means)
    print('1 \sigma lower', marge_stats_lower1sigma)
    print('1 \sigma upper', marge_stats_upper1sigma)
    print('2 \sigma lower', marge_stats_lower2sigma)
    print('2 \sigma upper', marge_stats_upper2sigma)
    print('3 \sigma lower', marge_stats_lower3sigma)
    print('3 \sigma upper', marge_stats_upper3sigma)

    best_stats = [x for x in str(samples.getLikeStats()).split(' ') if '.' in x]
    # best_params = [float(best_stats[3]), float(best_stats[8]), float(best_stats[13]), float(best_stats[18])]
    best_params = [float(best_stats[4]), float(best_stats[9]), float(best_stats[14]), float(best_stats[19]), float(best_stats[24]), float(best_stats[29]), float(best_stats[34]), float(best_stats[39]), float(best_stats[44]), float(best_stats[49])]

    plt.figure(1)
    X = ['b', 'theta', 'i', 'k', 'l', 'm', 'n', 'g', 'h', 'd']
    X_axis = np.arange(len(X))
    model_params = list(model_params)
    mcmc_params = list(best_params)
    plt.bar(X_axis-0.2, np.array(model_params[:-1])/np.array(model_params[:-1]), 0.2, label='Generated Data')
    plt.bar(X_axis+0.2, np.array(mcmc_params)/np.array(model_params[:-1]), 0.2, label='MCMC Data')
    plt.xticks(X_axis, X)
    plt.xlabel("Parameters")
    plt.ylabel("Parameters, normalised by generated data parameters")
    plt.title("MCMC Parameters Comparison")
    plt.legend()
    plt.savefig('Images/parameter_relative_ratios.png')

    plt.figure(2)
    new_states = []
    for i in range(len(hyperparams)):
        if i == 0 or i == 1 or i == 3:
            new_states.append(int(hyperparams[i]))
        else:
            new_states.append(hyperparams[i])
    t_span = np.array([0, new_states[-1]])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)
    plt.plot(t[:-1], real_infected, '.', label='Real Data')
    # plt.plot(t, generated_data_no_noise[1], label='Underlying Gen Data Signal')
    model_results_best = np.array(run_model('sirhd', best_params, new_states))
    infected_best = model_results_best[1]
    plt.plot(t, infected_best, label='MCMC Result')
    plt.xlabel('time')
    plt.ylabel('Infected People')
    plt.legend()
    plt.savefig('Images/results.png')

    for i in range(4):
        g = plots.get_single_plotter(width_inch=6)
        g.plot_1d(samples, 'p'+str(i+1), marker=best_params[i])
        g.export('./Images/p'+str(i+1)+'_dist.png')
    
    g = plots.getSubplotPlotter(width_inch=8)
    g.settings.alpha_filled_add=0.4
    g.settings.axes_fontsize = 20
    g.settings.lab_fontsize = 22
    g.settings.legend_fontsize = 20
    g.triangle_plot([samples], ['p1', 'p2','p3', 'p4', 'p5', 'p6', 'p7'], 
        filled_compare=True, 
        legend_labels=['Samples'], 
        legend_loc='upper right', 
        line_args=[{'ls':'-', 'color':'green'}], 
        contour_colors=['green'])
    g.export('./Images/Triangle.pdf')


    plt.figure(8)
    thetas = []
    for i in range(8):
        with open('./Chains/chain_'+str(i+1)+'.txt', 'r') as file:
            lines = file.readlines()
            thetas.append([float(lines[i].rstrip('\n').split(',')[1]) for i in range(len(lines))])
            x = np.linspace(1, len(thetas[i]), len(thetas[i]))
            plt.plot(x, thetas[i], label='p1 - Chain '+str(i+1))
    plt.legend()
    plt.xlabel('Number of Accepted Steps')
    plt.ylabel('P1')
    plt.savefig('Images/p1_evolution.png')

    plt.figure(9)
    thetas = []
    for i in range(8):
        with open('./Chains/chain_'+str(i+1)+'.txt', 'r') as file:
            lines = file.readlines()
            thetas.append([float(lines[i].rstrip('\n').split(',')[2]) for i in range(len(lines))])
            x = np.linspace(1, len(thetas[i]), len(thetas[i]))
            plt.plot(x, thetas[i], label='p2 - Chain '+str(i+1))
    plt.legend()
    plt.xlabel('Number of Accepted Steps')
    plt.ylabel('P2')
    plt.savefig('Images/p2_evolution.png')

    plt.figure(10)
    thetas = []
    for i in range(8):
        with open('./Chains/chain_'+str(i+1)+'.txt', 'r') as file:
            lines = file.readlines()
            thetas.append([float(lines[i].rstrip('\n').split(',')[3]) for i in range(len(lines))])
            x = np.linspace(1, len(thetas[i]), len(thetas[i]))
            plt.plot(x, thetas[i], label='p3 - Chain '+str(i+1))
    plt.legend()
    plt.xlabel('Number of Accepted Steps')
    plt.ylabel('P3')
    plt.savefig('Images/p3_evolution.png')

    plt.figure(11)
    thetas = []
    for i in range(8):
        with open('./Chains/chain_'+str(i+1)+'.txt', 'r') as file:
            lines = file.readlines()
            thetas.append([float(lines[i].rstrip('\n').split(',')[4]) for i in range(len(lines))])
            x = np.linspace(1, len(thetas[i]), len(thetas[i]))
            plt.plot(x, thetas[i], label='p4 - Chain '+str(i+1))
    plt.legend()
    plt.xlabel('Number of Accepted Steps')
    plt.ylabel('P4')
    plt.savefig('Images/p4_evolution.png')

    plt.figure(12)
    plt.plot(np.linspace(1, len(gelman_rubin), len(gelman_rubin)), gelman_rubin[:,0], label='p1')
    plt.plot(np.linspace(1, len(gelman_rubin), len(gelman_rubin)), gelman_rubin[:,1], label='p2')
    plt.plot(np.linspace(1, len(gelman_rubin), len(gelman_rubin)), gelman_rubin[:,2], label='p3')
    plt.plot(np.linspace(1, len(gelman_rubin), len(gelman_rubin)), gelman_rubin[:,3], label='p4')
    plt.plot(np.linspace(1, len(gelman_rubin), len(gelman_rubin)), np.linspace(1.05, 1.05, len(gelman_rubin)), '--', label='Convergence Criterion')
    plt.xlabel('Iteration')
    plt.ylabel('Gelman Rubin Statistic')
    plt.grid()
    plt.legend()
    plt.yscale('log')
    plt.savefig('Images/gelman_rubin.png')

    plt.show()
