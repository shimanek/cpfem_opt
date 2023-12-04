"""
Plots several figures per optimization:

1. Stress-strain curves of all parameter sets (per orientation, if applicable)
2. Stress-strain curve of best parameter set (per orientation, if applicable)
3. Convergence (lowest error as a function of iteration)
4. Histograms of parameter evaluations
5. Partial dependencies of the objective function

Prints best parameters to file out_best_params.txt
"""
import os
from skopt.plots import plot_evaluations, plot_objective

import numpy as np
from matmdl.optimizer import InOpt, load_opt, instantiate_optimizer
from matmdl.parser import uset

import matplotlib
matplotlib.use('Agg')  # backend selected for cluster compatibility
import matplotlib.pyplot as plt  # noqa: E402

#TODO: add Checkout(out, local=True) guard around outfile access

def main():
    orients = uset.orientations.keys()
    if __debug__: print('\n# start plotting')
    global in_opt
    in_opt = InOpt(uset.orientations, uset.params)
    fig0, ax0 = plt.subplots()
    labels0 = []
    colors0 = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ct_orient, orient in enumerate(orients):
        data = np.load(os.path.join(os.getcwd(), f'out_time_disp_force_{orient}.npy'))
        num_iter = len(data[0,0,:])
        print("DBG:PLOT:num_iter", num_iter)
        #-----------------------------------------------------------------------------------------------
        # plot all trials, in order:
        if __debug__: print('{}: all curves'.format(orient))
        fig, ax = plt.subplots()
        for i in range( num_iter ):
            eng_strain = data[:,1,i] / uset.length
            eng_stress = data[:,2,i] / uset.area
            ax.plot(eng_strain, eng_stress, alpha=0.2+(i+1)/num_iter*0.8,color='#696969')

        # plot experimental results:
        exp_filename = uset.orientations[orient]['exp']
        exp_SS = np.loadtxt(os.path.join(os.getcwd(), exp_filename), skiprows=1, delimiter=',')
        ax.plot(exp_SS[:,0], exp_SS[:,1], '-s',markerfacecolor='black', color='black', 
            label='Experimental ' + uset.grain_size_name)
        ax0.plot(exp_SS[:,0], exp_SS[:,1], 's',markerfacecolor=colors0[ct_orient], color='black', 
            label='Experimental ' + uset.grain_size_name)
        labels0.append(f"Exp. [{orient}]")

        # plot best guess:
        errors = np.loadtxt(os.path.join(os.getcwd(), 'out_progress.txt'), 
            skiprows=1, delimiter=',')[:,-1]
        loc_min_error = np.argmin(errors)
        eng_strain_best = data[:,1,loc_min_error] / uset.length
        eng_stress_best = data[:,2,loc_min_error] / uset.area
        ax.plot(eng_strain_best, eng_stress_best, '-o', alpha=1.0,color='blue', label='Best parameter set')
        ax0.plot(eng_strain_best, eng_stress_best, '-', alpha=1.0, linewidth=2, color=colors0[ct_orient], label='Best parameter set')
        labels0.append(f"Fit [{orient}]")

        plot_settings(ax)
        if uset.max_strain > 0:
            ax.set_xlim(right=uset.max_strain)
        fig.savefig(os.path.join(os.getcwd(), 'res_opt_' + orient + '.png'), 
            bbox_inches='tight', dpi=400)
        plt.close(fig)
        #-----------------------------------------------------------------------------------------------
        # print best paramters 
        params = np.loadtxt(os.path.join(os.getcwd(), 'out_progress.txt'), skiprows=1, delimiter=',')
        # ^ full list: 'iteration', 'Tau0', 'H0', 'TauS', 'hs', 'gamma0', 'error'
        best_params = [np.round(f,decimals=3) for f in params[loc_min_error,:]]
        with open('out_best_params.txt', 'w') as f:
            f.write('\nTotal iterations: ' + str(num_iter))
            f.write('\nBest iteration:   ' + str(int(best_params[0])))
            f.write('\nLowest error:     ' + str(best_params[-1]) + '\n')
            f.write('\nParameter names:\n' + ', '.join(in_opt.params) + '\n')
            f.write('Best parameters:\n' + ', '.join([str(f) for f in best_params[1:-1]]) + '\n\n')
            if len(uset.param_additional_legend) > 0:
                f.write('Fixed parameters:\n' + ', '.join(uset.param_additional_legend) + '\n')
                f.write('Fixed parameter values:\n' + ', '.join(
                    [str(get_param_value(f)) for f in uset.param_additional_legend]) + '\n\n')
        #-----------------------------------------------------------------------------------------------
        # plot best paramters
        legend_info = []
        for i, param in enumerate(in_opt.params):
            # 1st entry in best_params is iteration number, so use i+1
            legend_info.append(f'{name_to_sym(param)}={best_params[i+1]:.1f}')
        # also add additional parameters to legend:
        for param_name in uset.param_additional_legend:
            legend_info.append(name_to_sym(param_name) + '=' + str(get_param_value(param_name)))
        # add error value
        legend_info.append('Error: ' + str(best_params[-1]))
        legend_info = '\n'.join(legend_info)
        
        if __debug__: print('{}: best fit'.format(orient))
        fig, ax = plt.subplots()
        ax.plot(exp_SS[:,0], exp_SS[:,1], '-s',markerfacecolor='black', color='black', 
            label='Experimental ' + uset.grain_size_name)
        ax.plot(eng_strain_best, eng_stress_best, '-o', alpha=1.0,color='blue', label=legend_info)
        plot_settings(ax)
        if uset.max_strain > 0:
            ax.set_xlim(right=uset.max_strain)
        fig.savefig(os.path.join(os.getcwd(), 
            'res_single_' + orient + '.png'), bbox_inches='tight', dpi=400)
        plt.close(fig)

    # finish fig0, the plot of all sims and experimental data
    if len(orients) > 1:
        if __debug__: print('all stress-strain')
        plot_settings(ax0, legend=False)
        ax0.legend(loc='best', labels=labels0, fancybox=False)
        if uset.max_strain > 0:
            ax0.set_xlim(right=uset.max_strain)
        fig0.savefig(os.path.join(os.getcwd(), 'res_all.png'), bbox_inches='tight', dpi=400)
    else:
        plt.close(fig0)

    #-----------------------------------------------------------------------------------------------
    # plot convergence
    if __debug__: print('convergence information')
    fig, ax = plt.subplots()
    running_min = np.empty((num_iter))
    running_min[0] = errors[0]
    for i in range(1, num_iter):
        if errors[i] < running_min[i-1]:
            running_min[i] = errors[i]
        else:
            running_min[i] = running_min[i-1]
    ax.plot(list(range(num_iter)), running_min, '-o', color='blue')
    plot_settings(ax, legend=False)
    ax.set_xlabel('Iteration number')
    ax.set_ylabel('Lowest RMSE')
    fig.savefig('res_convergence.png', dpi=400, bbox_inches='tight')
    plt.close()
    #-----------------------------------------------------------------------------------------------
    # reload parameter guesses to use default plots
    opt = instantiate_optimizer(in_opt, uset)
    opt = load_opt(opt)
    # plot parameter distribution
    if __debug__: print('parameter evaluations')
    apply_param_labels(plot_evaluations(opt.get_result()), diag_label='Freq.')
    plt.savefig(fname='res_evaluations.png', bbox_inches='tight', dpi=600, transparent=False)
    plt.close()
    # plot partial dependence
    if __debug__: print('partial dependencies')
    apply_param_labels(plot_objective(opt.get_result()), diag_label='Objective')
    plt.savefig(fname='res_objective.png', bbox_inches='tight', dpi=600, transparent=False)
    plt.close()

    if __debug__: print('# stop plotting\n')

def plot_settings(ax, legend=True):
    ax.set_xlabel('Engineering Strain, m/m')
    ax.set_ylabel('Engineering Stress, MPa')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    if legend: ax.legend(loc='best', fancybox=False)
    plt.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_title(uset.title)

def apply_param_labels(ax_array, diag_label):
    shape = np.shape(ax_array)
    if len(shape) == 0: 
        ax_array.set_xlabel(name_to_sym(in_opt.params[0]))
        return
    for i in range(shape[0]):
        for j in range(shape[1]):
            ax_array[i,j].set_ylabel(name_to_sym(in_opt.params[i]))
            ax_array[i,j].set_xlabel(name_to_sym(in_opt.params[j]))

            if (i == j):  # diagonal subplots
                plt.setp(ax_array[i,j].get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')
                ax_array[i,j].set_ylabel(diag_label)
                ax_array[i,j].tick_params(axis='y', labelleft=False, labelright=True, left=False)
            
            if (j > 0) and not (i == j) and not (i == shape[0] - 1):  # middle section
                ax_array[i,j].tick_params(axis='y', left=False)
                ax_array[i,j].set_ylabel(None)
                ax_array[i,j].set_xlabel(None)
            
            if (i < shape[0] - 1) and not (i == j):  # not bottom row
                ax_array[i,j].tick_params(axis='x', bottom=False)
            
            if (i == shape[0] - 1) and not (i == j):  # bottom row
                plt.setp(ax_array[i,j].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

            if (i == shape[0] - 1) and not (i == j) and not (j == 0):  # middle bottom row
                ax_array[i,j].tick_params(axis='y', left=False)
                ax_array[i,j].set_ylabel(None)

    return ax_array


def name_to_sym(name, cap_sense=False):
    name_to_sym_dict = {
        'Tau0':r'$\tau_0$',
        'Tau01':r'$\tau_0^{(1)}$',
        'Tau02':r'$\tau_0^{(2)}$',
        'H0':r'$h_0$',
        'H01':r'$h_0^{(1)}$',
        'H02':r'$h_0^{(2)}$',
        'TauS':r'$\tau_s$',
        'TauS1':r'$\tau_s^{(1)}$',
        'TauS2':r'$\tau_s^{(2)}$',
        'q':r'$q$',
        'q1':r'$q_1$',
        'q2':r'$q_2$',
        'hs':r'$h_s$',
        'hs1':r'$h_s^{(1)}$',
        'hs2':r'$h_s^{(2)}$',
        'gamma0':r'$\gamma_0$',
        'gamma01':r'$\gamma_0^{(1)}$',
        'gamma02':r'$\gamma_0^{(2)}$',
        'g0': r'$\gamma_0$',
        'f0':r'$f_0$',
        'f1':r'$f_1$',
        'f2':r'$f_2$',
        'f3':r'$f_3$',
        'f4':r'$f_4$',
        'f5':r'$f_5$',
        'f01':r'$f_0^{(1)}$',
        'f02':r'$f_0^{(2)}$',
        'q0':r"$q_0$",
        'qA1':r'$q_{A1}$',
        'qB1':r'$q_{B1}$',
        'qA2':r'$q_{A2}$',
        'qB2':r'$q_{B2}$'
        }
    if cap_sense is True:
        have_key = name in name_to_sym_dict.keys()
    else:
        have_key = name.lower() in [key.lower() for key in name_to_sym_dict.keys()]
        name_to_sym_dict = {key.lower(): value for key, value in name_to_sym_dict.items()}
        name = name.lower()

    if have_key:
        return name_to_sym_dict[name]
    elif '_deg' in name:
        return name[:-4] + ' rot.'
    elif '_mag' in name:
        return name[:-4] + ' mag.'
    else:
        raise KeyError(f'Unknown parameter name: {name}')


def get_param_value(param_name):
    with open(uset.param_file, 'r') as f1:
        lines = f1.readlines()
    for line in lines:
        if line[:line.find('=')].strip() == param_name:
            return line[line.find('=')+1:].strip()


if __name__ == '__main__':
    main()
