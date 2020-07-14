'''
test script to optimize a single CPFEM parameter
Date: June 30, 2020
'''
import time
import os
import subprocess
import sys
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
from skopt import Optimizer
from skopt.plots import plot_convergence
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

### user input
param_list = ['Tau0', 'H0', 'TauS', 'hs', 'gamma0']
param_bounds = [ (1,30), (100,500), (1,200), (0,100), (0,0.4) ]
loop_len = 500
n_initial_points = 100
large_error = 5E4  # backup RMSE of runs which don't finish; first option uses 1.5 * max(RMSE)
exp_SS_file = 'expSS130um.txt'
### end input


# opt_progress = np.zeros( ( 1, len(param_list) + 1 ) )

def main():
    remove_out_files()

    global n_initial_points
    opt = Optimizer(
        dimensions = param_bounds, 
        base_estimator = 'gp',
        n_initial_points = n_initial_points
    )
    res = loop( opt, loop_len )
    plot_figs( res )

def loop(opt, loop_len):
    global opt_progress

    for i in range(loop_len):
        next_params = opt.ask()
        write_parameters(param_list, next_params)

        while param_check(param_list):  # True if Tau0 == TauS
            rmse = max_rmse(i)
            res = opt.tell( next_params, rmse )
            next_params = opt.ask()
            write_parameters(param_list, next_params)

        # submit job 
        os.system( 'abaqus job=UT_27grains user=umatcrystal_mod_XIT.f cpus=4 double int ask_delete=OFF' )
        time.sleep( 5 )

        if check_complete():
            # extract stress-strain
            os.system( 'abaqus cae nogui=extract_SS_singleEl.py' )

            # save stress-strain data
            combine_SS()

            # get error
            rmse = calc_error()  
            res = opt.tell( next_params, rmse )

            # save optimization progress
            if i == 0: opt_progress = np.transpose( np.asarray( [i, *next_params,rmse] ) )
            else:      opt_progress = np.vstack( (opt_progress, np.asarray( [i, *next_params,rmse] )) )
        else:
            rmse = max_rmse(i)
            res = opt.tell( next_params, rmse )
            if i == 0: opt_progress = np.transpose( np.asarray( [i, *next_params,rmse] ) )
            else:      opt_progress = np.vstack( (opt_progress, np.asarray( [i, *next_params,rmse] )) )

        with open('out_opt.pkl', 'wb') as f:
            pickle.dump(opt, f)

        opt_progress_header = ','.join( ['iteration'] + param_list + ['RMSE'] ) 
        np.savetxt('out_progress.txt',opt_progress, delimiter='\t', header=opt_progress_header)

    return res

def remove_out_files():
    out_files = [f for f in os.listdir(os.getcwd()) if f.startswith('out_')]
    if len(out_files) > 0:
        for f in out_files:
            os.remove(f)

def param_check(param_list):
    # True if tauS == tau0 
    if ('TauS' in param_list) or ('Tau0' in param_list):
        filename = [ f for f in os.listdir(os.getcwd()) if f.startswith('Mat_BW')][0]
        f1 = open( filename, 'r' )
        lines = f1.readlines()
        for line in lines:
            if line.startswith('Tau0'): tau0 = float( line[7:] )
            if line.startswith('TauS'): tauS = float( line[7:] )
        f1.close()
    return ( tau0 == tauS )

def max_rmse(loop_number):
    global large_error
    if loop_number > 10:
        return max(opt_progress[:,-1]) * 1.5
    else:
        return large_error

def check_complete():
    stafile = [ f for f in os.listdir(os.getcwd()) if f.startswith( 'UT' ) ][0]
    stafile = stafile[ 0 : stafile.find('.') ] + '.sta'
    if os.path.isfile( stafile ):
            last_line = str( subprocess.check_output( ['tail', '-1', stafile] ) )
    else: 
        last_line = ''
    return ( 'SUCCESSFULLY' in last_line )

def combine_SS():
    filename = 'out_time_disp_force.npy'
    sheet = np.loadtxt( 'allArray.csv', delimiter=',', skiprows=1 )
    if os.path.isfile( filename ): 
        dat = np.load( filename )
        dat = np.dstack( (dat,sheet) )
    else:
        dat = sheet
    np.save( filename, dat )

def calc_error():
    global exp_SS_file
    simSS = np.loadtxt( 'allArray.csv', delimiter=',', skiprows=1 )[:,1:]
    # TODO get simulation dimensions at beginning of running this file, pass to this function
    simSS[:,0]/3  # disp to strain
    simSS[:,1]/(3**2)  # force to stress

    # smooth out simulated SS
    smoothenedSS = interp1d( simSS[:,0], simSS[:,1] )

    # load experimental data
    expSS = np.loadtxt( exp_SS_file, skiprows=2 )

    # error function
    deviations = np.asarray( [ smoothenedSS(expSS[i,0]) - expSS[i,1] for i in range(len(expSS)) ] )
    rmse = np.sqrt( np.sum( deviations**2 ) / len(expSS) ) 

    return rmse

def write_parameters(param_list, next_params):
    filename = [ f for f in os.listdir(os.getcwd()) if f.startswith('Mat_BW')][0]
    f1 = open( filename, 'r' )
    f2 = open( 'temp_mat_file.inp', 'w+' )
    lines = f1.readlines()
    for line in lines:
        skip = False
        for param in param_list:
            if line.startswith(param):
                f2.write( param + ' = ' + str(next_params[param_list.index(param)]) + '\n')
                skip = True
        if not skip:
            f2.write( line )
    f1.close()
    f2.close()

    os.remove( filename )
    os.rename( 'temp_mat_file.inp', filename )

def plot_figs(res):
    plot_convergence(res)
    plt.savefig('out_convergence.png',dpi=400)
    plt.close()

if __name__ == '__main__':
    main()
