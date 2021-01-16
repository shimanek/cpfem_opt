'''
Date: 2020-04-15
Quickly plot a comparison of the stress-strain response
of multiple crystal plasticity runs output 
from the optimization procedure.
Plots 3 figures per subfolder: all params, best params, convergence.
Prints best parameters.
Last Mod: 2020-12-07
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import opt_input as uset

# TODO use inputs from opt_input.py for all of this 
# TODO use param_list to determine which param values to show in legend 
### user input 
fixed_params = []  # usu. [Tau0_value, h0_value]
### end input

def main():
    data = np.load( os.path.join(os.getcwd(), 'out_time_disp_force.npy') )
    num_iter = len(data[0,0,:])
    #-----------------------------------------------------------------------------------------------
    # plot all trials, in order:
    fig, ax = plt.subplots()
    for i in range( num_iter ):
        eng_strain = data[:,1,i] / uset.length
        eng_stress = data[:,2,i] / uset.area
        ax.plot(eng_strain, eng_stress, alpha=0.2+i/num_iter/0.8,color='#696969')

    # plot experimental results:
    exp_filename = [f for f in os.listdir(os.getcwd()) \
        if f.startswith('exp') and f.endswith('.csv')][0]
    exp_SS = np.loadtxt(os.path.join(os.getcwd(), exp_filename), skiprows=1, delimiter=',')
    ax.plot(exp_SS[:,0], exp_SS[:,1], '-s',markerfacecolor='black', color='black', 
        label='Experimental ' + grain_size_name + 'um')

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
        'res_opt_' + uset.grain_size_name + 'um.png'), bbox_inches='tight', dpi=400)
    plt.close()
    #-----------------------------------------------------------------------------------------------
    # print best paramters 
    params = np.loadtxt(os.path.join(os.getcwd(), 'out_progress.txt'), skiprows=1, delimiter='\t')
    # ^ full list: 'iteration', 'Tau0', 'H0', 'TauS', 'hs', 'gamma0', 'error'
    best_params = [np.round(f,decimals=2) for f in params[loc_min_error,:]]
    for param in fixed_params[::-1]:
        best_params = [best_params[0]] + [param] + best_params[1:]
    with open('out_best_params.txt', 'w') as f:
        f.write('Total iterations: ' + str(num_iter) + '\n')
        f.write('Best parameters:\n')  # TODO write which parameters they are in separate line
        f.write(str(list(best_params)) + '\n')
    #-----------------------------------------------------------------------------------------------
    # plot best paramters 
    legend_info = r'$\tau_0=$'   + str(best_params[1])  + '\n' + \
                  r'$h_0=$'      + str(best_params[2])  + '\n' + \
                  r'$\tau_s=$'   + str(best_params[3])  + '\n' + \
                  r'$h_s=$'      + str(best_params[4])  + '\n' + \
                  r'$\gamma_0=$' + str(best_params[5])  
                #   r', $f_0=$'      + best_params[i] + \
                #   r', $q=$'        + best_params[i]
    fig, ax = plt.subplots()
    ax.plot(exp_SS[:,0], exp_SS[:,1], '-s',markerfacecolor='black', color='black', 
        label='Experimental ' + uset.grain_size_name + 'um')
    ax.plot(eng_strain_best, eng_stress_best, '-o', alpha=1.0,color='blue', label=legend_info)
    plot_settings()
    plt.savefig(os.path.join(os.getcwd(), 
        'res_single_' + uset.grain_size_name + 'um.png'), bbox_inches='tight', dpi=400)
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

if __name__ == '__main__':
    main()

