'''
test script to optimize CPFEM parameters
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
param_bounds = [ (1,100), (100,500), (1,200), (0,100), (0,0.4) ]
loop_len = 500
n_initial_points = 100
large_error = 5e3  # backup RMSE of runs which don't finish; first option uses 1.5 * IQR(first few RMSE)
exp_SS_file = [f for f in os.listdir() if f.startswith('exp')][0]
length = 9
area = 9*9
jobname = 'UT_729grains'
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
    load_opt(opt)
    res = loop( opt, loop_len )
    plot_figs( res )

def loop(opt, loop_len):
    global opt_progress

    for i in range(loop_len):

        if i==0: get_first()

        next_params = opt.ask()
        write_parameters(param_list, next_params)

        if param_check(param_list):  # True if Tau0 >= TauS
            # TODO add sheet of zeros to out_time_disp_force.npy (implemented below but needs cleaning up!)
            rmse = max_rmse(i)
            res = opt.tell( next_params, rmse )
            next_params = opt.ask()
            write_parameters(param_list, next_params)
            combine_SS(zeros=True)
            # TODO just make the first row of opt_progress zeros and delete it after the last step 
            if i == 0: opt_progress = np.transpose( np.asarray( [i, *next_params,rmse] ) )
            else:      opt_progress = np.vstack( (opt_progress, np.asarray( [i, *next_params,rmse] )) )
        else:
            # submit job 
            os.system( 'abaqus job=' + jobname + ' user=umatcrystal_mod_XIT.f cpus=8 double int ask_delete=OFF' )
            time.sleep( 5 )

            if not check_complete():
                refine_run()

            if check_complete():
                # extract data to temp_time_disp_force.csv
                os.system( 'abaqus python -c "from opt_extract import write2file; write2file()"' )

                # save stress-strain data
                combine_SS(zeros=False)

                # get error
                rmse = calc_error()  
                res = opt.tell( next_params, rmse )

                # save optimization progress
                if i == 0: opt_progress = np.transpose( np.asarray( [i, *next_params,rmse] ) )
                else:      opt_progress = np.vstack( (opt_progress, np.asarray( [i, *next_params,rmse] )) )
            else:
                rmse = max_rmse(i)
                res = opt.tell( next_params, rmse )
                combine_SS(zeros=True)
                if i == 0: opt_progress = np.transpose( np.asarray( [i, *next_params,rmse] ) )
                else:      opt_progress = np.vstack( (opt_progress, np.asarray( [i, *next_params,rmse] )) )

        opt_progress_header = ','.join( ['iteration'] + param_list + ['RMSE'] ) 
        np.savetxt('out_progress.txt',opt_progress, delimiter='\t', header=opt_progress_header)

    return res

def get_first():
    os.system( 'abaqus job=' + jobname + ' user=umatcrystal_mod_XIT.f cpus=8 double int ask_delete=OFF' )
    time.sleep(5)
    have_1st = check_complete()
    if have_1st: 
        os.system( 'abaqus python -c "from opt_extract import write2file; write2file()"' )
    else: 
        refine_run()
        time.sleep(5)
        os.system( 'abaqus python -c "from opt_extract import write2file; write2file()"' )


def load_opt(opt):
    in_filename = 'in_opt.txt'
    if os.path.isfile(in_filename):
        prev_data = np.loadtxt(in_filename, skiprows=1)
        x_in = prev_data[:,1:-1].tolist()
        y_in = prev_data[:,-1].tolist()
        opt.tell(x_in, y_in)

def remove_out_files():
    out_files = [f for f in os.listdir(os.getcwd()) if f.startswith('out_')]
    if len(out_files) > 0:
        for f in out_files:
            os.remove(f)

def param_check(param_list):
    # True if tau0 >= tauS, which is bad
    if ('TauS' in param_list) or ('Tau0' in param_list):
        filename = [ f for f in os.listdir(os.getcwd()) if f.startswith('Mat_BW')][0]
        f1 = open( filename, 'r' )
        lines = f1.readlines()
        for line in lines:
            if line.startswith('Tau0'): tau0 = float( line[7:] )
            if line.startswith('TauS'): tauS = float( line[7:] )
        f1.close()
    return ( tau0 >= tauS )

def max_rmse(loop_number):
    global large_error
    grace = 15
    if loop_number < grace:
        return large_error
    elif loop_number >= grace:
        errors = np.delete( opt_progress[:grace,-1], np.where( opt_progress[:grace,-1] == large_error ) )
        if len(errors) < np.round( grace/2 ):
            return large_error
        else:
            iq1, iq3 = np.quantile(errors, [0.25,0.75])
            return (iq3-iq1)*1.5

def check_complete():
    stafile = [ f for f in os.listdir(os.getcwd()) if f.startswith( 'UT' ) ][0]
    stafile = stafile[ 0 : stafile.find('.') ] + '.sta'
    if os.path.isfile( stafile ):
            last_line = str( subprocess.check_output( ['tail', '-1', stafile] ) )
    else: 
        last_line = ''
    return ( 'SUCCESSFULLY' in last_line )

def refine_run():
    """
    cut max increment size by `factor`
    """
    factor = 5
    # remove old lock file from previous unfinished simulation
    os.system('rm *.lck')
    # find input file TODO put main input file name up top, not hardcoded as here
    filename = [ f for f in os.listdir(os.getcwd()) if f.startswith('UT') and f.endswith('.inp')][0]
    with open(filename, 'r') as f:
        lines = f.readlines()
    # find line after step line:
    step_line_ind = [ i for i, line in enumerate(lines) if line.lower().startswith('*static')][0] + 1 
    step_line = lines[step_line_ind].strip().split(', ')
    original_increment = float(step_line[-1])
    # use original 
    new_step_line = step_line[:-1] + [ '%.4f' % (original_increment/factor) ] 
    new_step_line_str = str(new_step_line[0])
    for i in range(1, len(new_step_line)):
        new_step_line_str = new_step_line_str + ', '
        new_step_line_str = new_step_line_str + str(new_step_line[i])
    new_step_line_str = new_step_line_str + '\n'
    with open(filename, 'w') as f:
        f.writelines(lines[:step_line_ind])
        f.writelines(new_step_line_str)
        f.writelines(lines[step_line_ind+1:])
    os.system( 'abaqus job=' + jobname + ' user=umatcrystal_mod_XIT.f cpus=8 double int ask_delete=OFF' )
    if check_complete():
        with open(filename, 'w') as f:
            f.writelines(lines)
    else:
        refine_run()

def combine_SS(zeros:bool):
    # TODO problems here: incomplete runs throw error, derail entire job 
    filename = 'out_time_disp_force.npy'
    sheet = np.loadtxt( 'temp_time_disp_force.csv', delimiter=',', skiprows=1 ) #TODO what if allarray does not exist? how to get shape for zeros? (maybe from input file)
    if zeros:
        sheet = np.zeros( (np.shape(sheet)) )
    if os.path.isfile( filename ): 
        dat = np.load( filename )
        dat = np.dstack( (dat,sheet) )
        # error: along dimension 0, the array at index 0 has size 18 and the array at index 1 has size 101
        # so sheet dimension is correct 
    else:
        dat = sheet
    np.save( filename, dat )

def calc_error():
    global exp_SS_file
    global length
    global area
    simSS = np.loadtxt( 'temp_time_disp_force.csv', delimiter=',', skiprows=1 )[:,1:]
    # TODO get simulation dimensions at beginning of running this file, pass to this function
    simSS[:,0] = simSS[:,0]/length  # disp to strain
    simSS[:,1] = simSS[:,1]/area    # force to stress

    # load experimental data
    expSS = np.loadtxt( exp_SS_file, skiprows=2 )

    # deal with unequal data lengths 
    if simSS[-1,0] >= expSS[-1,0]:
        # chop off simSS
        cutoff = np.where( simSS[:,0] > expSS[-1,0] )[0][0] - 1
        simSS = simSS[:cutoff,:]
        cutoff_strain = simSS[-1,0]
    else:
        # chop off expSS
        cutoff = np.where(simSS[-1,0] < expSS[:,0])[0][0] - 1
        expSS = expSS[:cutoff,:]
        cutoff_strain = expSS[-1,0]

    # smooth out simulated SS
    smoothedSS = interp1d( simSS[:,0], simSS[:,1] )

    smoothedExp = interp1d( expSS[:,0], expSS[:,1] )
    num_error_eval_pts = 1000
    x_error_eval_pts = np.linspace( expSS[0,0], cutoff_strain, num = num_error_eval_pts )
    fineSS = smoothedExp( x_error_eval_pts )
    # strictly limit to interpolation
    while x_error_eval_pts[-1] >= expSS[-1,0]:
        fineSS = np.delete(fineSS, -1)
        x_error_eval_pts = np.delete(x_error_eval_pts, -1)

    # error function
    deviations = np.asarray( [ smoothedSS( x_error_eval_pts[i] ) - fineSS[i] for i in range(len(fineSS)) ] )
    rmse = np.sqrt( np.sum( deviations**2 ) / len(fineSS) ) 

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
