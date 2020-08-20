'''
April 15, 2020
quickly plot a comparison of the stress-strain response
of 137 grain polycrystalline model for PAN h_0 parameter
Modified May 19 to look at all parameters 
Modified Aug 20 to look at single crystal results
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
# -------------------------------------------------------------------------------------------------
# inputs:
title = 'Single Crystal (Haasen [128])'
data_folder = 'in_data'
axial_length = 15
transverse_length = 1
experimental_file = 'haasen_1-2-8_direction.txt'
exp_file_skiprows = 1
run_on_cluster = False
# -------------------------------------------------------------------------------------------------
# functions 
def filename(title):
    filename = ''
    i = 0
    for char in title:
        if char == ' ':
            filename += '_'
        elif char.isalnum():
            filename += char
        i += 1
    return filename.lower()
def plot_tuning(ax):
    ax.set_xlabel('Engineering Strain, m/m')
    ax.set_ylabel('Engineering Stress, MPa')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc='best')
def cluster_start():
    os.system('mkdir ' + data_folder + '; mv out_* ' + data_folder)
    os.system('cp ' + experimental_file + ' ' + data_folder)
# -------------------------------------------------------------------------------------------------
# load data 
if run_on_cluster: cluster_start()
data = np.load( os.path.join(data_folder, 'out_time_disp_force.npy') )
data_num = len(data[0,0,:])
params = np.loadtxt(os.path.join(data_folder, 'out_progress.txt'),
                    skiprows=1, delimiter='\t')
# ^ cols: 'iteration', 'Tau0', 'H0', 'TauS', 'hs', 'gamma0', 'error'
loc_min_error = np.argmin(params[:,-1])
best_params = [np.round(f,decimals=2) for f in params[loc_min_error,:]]
print('\nLowest RMSE:  ', best_params[-1], '\n')
# load all tries:
time_disp_force = np.load( os.path.join(data_folder, 'out_time_disp_force.npy') )
time_disp_force[:,1,:] = time_disp_force[:,1,:]/axial_length            # strain
time_disp_force[:,2,:] = time_disp_force[:,2,:]/(transverse_length**2)  # stress
# -------------------------------------------------------------------------------------------------
# plot 
fig, ax = plt.subplots()
# legend
legend_info = r'$\tau_0=$'    + str(best_params[1])  + '\n' + \
                r'$h_0=$'      + str(best_params[2])  + '\n' + \
                r'$\tau_s=$'   + str(best_params[3])  + '\n' + \
                r'$h_s=$'      + str(best_params[4])  + '\n' + \
                r'$\gamma_0=$' + str(best_params[5])  
            #   r', $f_0=$'      + best_params[i] + \
            #   r', $q=$'        + best_params[i]

# plot experimental results:
exp_SS = np.loadtxt(os.path.join(os.getcwd(), data_folder, experimental_file),
                    skiprows=exp_file_skiprows)
ax.plot(exp_SS[:,0], exp_SS[:,1], '-s',markerfacecolor='black', color='black',
        label='Experimental')

# plot best guess:
eng_strain_best = data[:,1,loc_min_error] / axial_length
eng_stress_best = data[:,2,loc_min_error] / transverse_length**2
ax.plot(eng_strain_best, eng_stress_best, '-o', alpha=1.0,color='blue', label=legend_info)
plot_tuning(ax)
ax.set_title(title)
plt.savefig(os.path.join(os.getcwd(), 'res_' + filename(title) + '_best.png'),
            bbox_inches='tight', dpi=400)

# plot all guesses
fig, ax = plt.subplots()
for i in range( data_num ):
    eng_strain = data[:,1,i] / axial_length
    eng_stress = data[:,2,i] / transverse_length**2
    ax.plot(eng_strain, eng_stress, alpha=0.2+i/data_num/0.8,color='#696969')
ax.plot(exp_SS[:,0], exp_SS[:,1], '-s',markerfacecolor='black', color='black',
        label='Experimental')
ax.plot(eng_strain_best, eng_stress_best, '-o', alpha=1.0,color='blue', label=legend_info)
plot_tuning(ax)
plt.savefig(os.path.join(os.getcwd(), 'res_' + filename(title) + '_all.png'),
            bbox_inches='tight', dpi=400)
