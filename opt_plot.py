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

def main():
    data = np.load( os.path.join(os.getcwd(), 'out_time_disp_force.npy') )
    num_iter = len(data[0,0,:])
    #-----------------------------------------------------------------------------------------------
    # plot all trials, in order:
    fig, ax = plt.subplots()
    for i in range( num_iter ):
        eng_strain = data[:,1,i] / uset.length
        eng_stress = data[:,2,i] / uset.area
        ax.plot(eng_strain, eng_stress, alpha=0.2+(i+1)/num_iter*0.8,color='#696969')

    # plot experimental results:
    exp_filename = 'temp_expSS.csv' if (float(uset.max_strain) == 0.0) else uset.exp_SS_file
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
    plt.savefig(os.path.join(os.getcwd(), 
        'res_opt_' + uset.grain_size_name + '.png'), bbox_inches='tight', dpi=400)
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
        f.write('\nParameter names:\n' + ', '.join(uset.param_list) + '\n')
        f.write('Best parameters:\n' + ', '.join([str(f) for f in best_params[1:-1]]) + '\n\n')
        if len(uset.param_additional_legend) > 0:
            f.write('Fixed parameters:\n' + ', '.join(uset.param_additional_legend) + '\n')
            f.write('Fixed parameter values:\n' + ', '.join(
                [str(get_param_value(f)) for f in uset.param_additional_legend]) + '\n\n')
    #-----------------------------------------------------------------------------------------------
    # plot best paramters 
    name_to_sym = {
        'Tau0':r'$\tau_0$',
        'H0':r'$h_0$',
        'TauS':r'$\tau_s$',
        'hs':r'$h_s$', 
        'gamma0':r'$\gamma_0$',
        'f0':r'$f_0$',
        'q':r'$q$'}
    legend_info = []
    for i, param in enumerate(uset.param_list):
        # 1st entry in best_params is iteration number, so use i+1
        legend_info.append( name_to_sym[param] + '=' + str(best_params[i+1]))
    # also add additional parameters to legend:
    for param_name in uset.param_additional_legend:
        legend_info.append( name_to_sym[param_name] + '=' + str(get_param_value(param_name)))
    # add error value
    legend_info.append('Error: ' + str(best_params[-1]))
    legend_info = '\n'.join(legend_info)
    
    fig, ax = plt.subplots()
    ax.plot(exp_SS[:,0], exp_SS[:,1], '-s',markerfacecolor='black', color='black', 
        label='Experimental ' + uset.grain_size_name)
    ax.plot(eng_strain_best, eng_stress_best, '-o', alpha=1.0,color='blue', label=legend_info)
    plot_settings()
    plt.savefig(os.path.join(os.getcwd(), 
        'res_single_' + uset.grain_size_name + '.png'), bbox_inches='tight', dpi=400)
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


def get_param_value(param_name):
    with open(uset.param_file, 'r') as f1:
        lines = f1.readlines()
    for line in lines:
        if line[:line.find('=')].strip() == param_name:
            return line[line.find('=')+1:].strip()


if __name__ == '__main__':
    main()
