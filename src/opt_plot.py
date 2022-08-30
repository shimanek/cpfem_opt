"""
Plots 3 figures: 
1. Stress-strain curves of all parameter sets
2. Stress-strain curve of best parameter set
3. Convergence (lowest error as a function of iteration)
Prints best parameters to file out_best_params.txt
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import opt_input as uset
from opt_fea import InOpt

def main(orient):
    in_opt = InOpt(uset.orientations, uset.param_list, uset.param_bounds)
    data = np.load(os.path.join(os.getcwd(), f'out_time_disp_force_{orient}.npy'))
    num_iter = len(data[0,0,:])
    #-----------------------------------------------------------------------------------------------
    # plot all trials, in order:
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

    # plot best guess:
    errors = np.loadtxt(os.path.join(os.getcwd(), 'out_progress.txt'), 
        skiprows=1, delimiter='\t')[:,-1]
    loc_min_error = np.argmin(errors)
    eng_strain_best = data[:,1,loc_min_error] / uset.length
    eng_stress_best = data[:,2,loc_min_error] / uset.area
    ax.plot(eng_strain_best, eng_stress_best, '-o', alpha=1.0,color='blue', label='Best parameter set')

    # plot tuning:
    def plot_settings():
        ax.set_xlabel('Engineering Strain, m/m')
        ax.set_ylabel('Engineering Stress, MPa')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(loc='best')
        plt.tick_params(which='both', direction='in', top=True, right=True)
        ax.set_title(uset.title)

    plot_settings()
    plt.savefig(os.path.join(os.getcwd(), 'res_opt_' + orient + '.png'), 
        bbox_inches='tight', dpi=400)
    plt.close()
    #-----------------------------------------------------------------------------------------------
    # print best paramters 
    params = np.loadtxt(os.path.join(os.getcwd(), 'out_progress.txt'), skiprows=1, delimiter='\t')
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
    def name_to_sym(name):
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
            'f0':r'$f_0$',
            'f01':r'$f_0^{(1)}$',
            'f02':r'$f_0^{(2)}$',
            'qA1':r'$q_{A1}$',
            'qB1':r'$q_{B1}$',
            'qA2':r'$q_{A2}$',
            'qB2':r'$q_{B2}$'
            }
        if name in name_to_sym_dict.keys():
            return name_to_sym_dict[name]
        elif '_deg' in name:
            return name[:-4] + ' rot.'
        elif '_mag' in name:
            return name[:-4] + ' mag.'
        else:
            raise KeyError('Uknown parameter name')


    legend_info = []
    for i, param in enumerate(in_opt.params):
        # 1st entry in best_params is iteration number, so use i+1
        legend_info.append(name_to_sym(param) + '=' + str(best_params[i+1]))
    # also add additional parameters to legend:
    for param_name in uset.param_additional_legend:
        legend_info.append(name_to_sym(param_name) + '=' + str(get_param_value(param_name)))
    # add error value
    legend_info.append('Error: ' + str(best_params[-1]))
    legend_info = '\n'.join(legend_info)
    
    fig, ax = plt.subplots()
    ax.plot(exp_SS[:,0], exp_SS[:,1], '-s',markerfacecolor='black', color='black', 
        label='Experimental ' + uset.grain_size_name)
    ax.plot(eng_strain_best, eng_stress_best, '-o', alpha=1.0,color='blue', label=legend_info)
    plot_settings()
    plt.savefig(os.path.join(os.getcwd(), 
        'res_single_' + orient + '.png'), bbox_inches='tight', dpi=400)
    plt.close()
    #-----------------------------------------------------------------------------------------------
    # plot convergence
    fig, ax = plt.subplots()
    running_min = np.empty((num_iter))
    running_min[0] = errors[0]
    for i in range(1, num_iter):
        if errors[i] < running_min[i-1]:
            running_min[i] = errors[i]
        else:
            running_min[i] = running_min[i-1]
    ax.plot(list(range(num_iter)), running_min, '-o', color='blue')
    plot_settings()
    ax.set_xlabel('Iteration number')
    ax.set_ylabel('Lowest RMSE')
    fig.savefig('res_convergence.png', dpi=400, bbox_inches='tight')
    plt.close()
    #-----------------------------------------------------------------------------------------------
    # plot parameter distribution


def get_param_value(param_name):
    assert param_name in uset.param_list, \
        "Error: can only add extra material parameters."
    with open(uset.param_file, 'r') as f1:
        lines = f1.readlines()
    for line in lines:
        if line[:line.find('=')].strip() == param_name:
            return line[line.find('=')+1:].strip()


if __name__ == '__main__':
    for orient in uset.orientations.keys():
        main(orient=orient)
