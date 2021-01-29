"""
Script to optimize CPFEM parameters using as the engine Abaqus and Huang's subroutine.
All user inputs should be in `opt_input.py` file.
Date: June 30, 2020
"""
if __name__ == '__main__':
    import time
    import os
    import subprocess
    import sys
    import numpy as np
    from skopt import Optimizer
    from scipy.interpolate import interp1d
else:
    from odbAccess import *
    from abaqusConstants import *
    from odbMaterial import *
    from odbSection import *
import opt_input as uset  # user settings file

def main():
    remove_out_files()
    set_strain_inp()
    # global n_initial_points
    opt = Optimizer(
        dimensions = uset.param_bounds, 
        base_estimator = 'gp',
        n_initial_points = uset.n_initial_points)
    load_opt(opt)

    loop( opt, uset.loop_len )

def loop(opt, loop_len):
    get_first()
    for i in range(loop_len):
        def single_loop(opt, i):
            global opt_progress  # global progress tracker, row:(i, params, error)
            next_params = opt.ask()  # get parameters to test
            write_parameters(next_params)  # write params to file

            while param_check(uset.param_list):  # True if Tau0 >= TauS
                # this tells opt that params are bad but does not record it elsewhere
                opt.tell( next_params, max_rmse(i) )
                next_params = opt.ask()
                write_parameters(next_params)
            else:
                job_run()
                if not check_complete():  
                # try decreasing max increment size
                    refine_run()  
                if not check_complete():  
                # if it still fails, write max_rmse, go to next parameterset
                    write_maxRMSE(i, next_params, opt)
                    return
                else:
                    job_extract()  # extract data to temp_time_disp_force.csv
                    combine_SS(zeros=False)  # save stress-strain data
                    rmse = calc_error()  # get error
                    opt.tell( next_params, rmse )
                    opt_progress = update_progress(i, next_params, rmse)
                    write_opt_progress()
        single_loop(opt, i)

def set_strain_inp():
    """
    Modify displacement B.C. in main Abaqus input file to match max strain.
    Modify global experimental data filename to point to truncated data.
    """
    global exp_SS_file

    # limit experimental data to within max_strain
    expSS = np.loadtxt( uset.exp_SS_file, skiprows=1, delimiter=',' )
    expSS = expSS[expSS[:,0].argsort()]

    if float(uset.max_strain) == 0.0:
        max_strain = max(np.loadtxt( uset.exp_SS_file, skiprows=1, delimiter=',' )[:,0])
    else:
        max_strain = uset.max_strain
        max_point = 0
        while expSS[max_point,0] <= max_strain:
            max_point += 1
        expSS = expSS[:max_point, :]
    np.savetxt('temp_expSS.csv', expSS, delimiter=',')
    exp_SS_file = 'temp_expSS.csv'

    # input file:
    max_bound = round(max_strain * uset.length, 4) #round to 4 digits

    filename = uset.jobname + '.inp'
    with open(filename, 'r') as f:
        lines = f.readlines()

    # find last number after RP-TOP under *Boundary
    bound_line_ind = [ i for i, line in enumerate(lines) \
        if line.lower().startswith('*boundary')][0] + 4
    bound_line = [ number.strip() for number in lines[bound_line_ind].strip().split(',') ]

    new_bound_line = bound_line[:-1] + [ max_bound ] 
    new_bound_line_str = str(new_bound_line[0])

    for i in range(1, len(new_bound_line)):
        new_bound_line_str = new_bound_line_str + ', '
        new_bound_line_str = new_bound_line_str + str(new_bound_line[i])
    new_bound_line_str = new_bound_line_str + '\n'

    # write to UT_729grains.inp
    with open(filename, 'w') as f:
        f.writelines(lines[:bound_line_ind])
        f.writelines(new_bound_line_str)
        f.writelines(lines[bound_line_ind+1:])

def write_opt_progress():
    global opt_progress
    opt_progress_header = ','.join( ['iteration'] + uset.param_list + ['RMSE'] ) 
    np.savetxt('out_progress.txt',opt_progress, delimiter='\t', header=opt_progress_header)

def update_progress(i, next_params, rmse):
    global opt_progress
    if i == 0: opt_progress = np.transpose( np.asarray( [i] + next_params + [rmse] ) )
    else: opt_progress = np.vstack( (opt_progress, np.asarray( [i] + next_params + [rmse] )) )
    return opt_progress

def write_maxRMSE(i, next_params, opt):
    global opt_progress
    rmse = max_rmse(i)
    opt.tell( next_params, rmse )
    combine_SS(zeros=True)
    opt_progress = update_progress(i, next_params, rmse)
    write_opt_progress()

def job_run():
    os.system( 'abaqus job=' + uset.jobname + \
        ' user=umatcrystal_mod_XIT.f cpus=8 double int ask_delete=OFF' )
    time.sleep( 5 )

def job_extract():
    os.system( 'abaqus python -c "from opt_fea import write2file; write2file()"' )

def get_first():
    """
    Run one simulation, using its output dimensions to get shape of output data.
    """
    job_run()
    have_1st = check_complete()
    if not have_1st: 
        refine_run()
    job_extract()

def load_opt(opt):
    """
    Load input files of previous optimizations to use as initial points in current optimization.
    Note the parameter bounds for `in`-files must be within current bounds.
    """
    filename = 'in_opt.txt'
    arrayname = 'in_opt.npy'
    if os.path.isfile(filename):
        prev_data = np.loadtxt(filename, skiprows=1)
        x_in = prev_data[:,1:-1].tolist()
        y_in = prev_data[:,-1].tolist()
        opt.tell(x_in, y_in)
    if os.path.isfile(arrayname):
        np.save(arrayname, np.load(arrayname))

def remove_out_files():
    out_files = [f for f in os.listdir(os.getcwd()) if f.startswith('out_')]
    if len(out_files) > 0:
        for f in out_files:
            os.remove(f)

def param_check(param_list):
    """
    True if tau0 >= tauS, which is bad, not practically but in theory.
    In theory, tau0 should always come before tauS.
    """
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
    """
    Return an estimate of a "large" error value to dissuade the optimizer from repeating
    areas in parameter space where Abaqus+UMAT can't complete calculations.
    Often this ends up defaulting to `uset.large_error`.
    """
    grace = 15
    if loop_number < grace:
        return uset.large_error
    elif loop_number >= grace:
        errors = np.delete( opt_progress[:grace,-1], \
            np.where( opt_progress[:grace,-1] == uset.large_error ) )
        if len(errors) < np.round( grace/2 ):
            return uset.large_error
        else:
            iq1, iq3 = np.quantile(errors, [0.25,0.75])
            return (iq3-iq1)*1.5

def check_complete():
    """
    Return true if Abaqus has finished sucessfully.
    """
    stafile = uset.jobname + '.sta'
    if os.path.isfile( stafile ):
            last_line = str( subprocess.check_output( ['tail', '-1', stafile] ) )
    else: 
        last_line = ''
    return ( 'SUCCESSFULLY' in last_line )

def refine_run(ct=0):
    """
    Cut max increment size by `factor`, possibly multiple times (up to 
    `uset.recursion_depth` or until Abaqus finished successfully).
    """
    # TODO fails if there is more than one load step!
    # can get current load step from first col, 3rd-to-last row of sta file
    sta_file = [ f for f in os.listdir(os.getcwd()) if f.endswith('.sta')][0]
    with open(sta_file,'r') as f:
        last_line = f.readlines()[-3]
    step_current = [int(char) for char in list(last_line) if char.isnumeric()][0]
    factor = 5.0
    ct += 1
    # remove old lock file from previous unfinished simulation
    os.system('rm *.lck')
    # find input file TODO put main input file name up top, not hardcoded as here
    filename = uset.jobname + '.inp'
    tempfile = 'temp_input.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # exit strategy:
    if ct == 1:  # need to save original parameters outside of this recursive function
        with open(tempfile, 'w') as f:
            f.writelines(lines)
    def write_original(filename):
        with open(tempfile, 'r') as f:
            lines = f.readlines()
        with open(filename, 'w') as f:
            f.writelines(lines)

    # find line after step line:
    step_line_ind = [ i for i, line in enumerate(lines) \
        if line.lower().startswith('*static')] \
        [step_current - 1] + 1 
    step_line = [ number.strip() for number in lines[step_line_ind].strip().split(',') ]
    original_increment = float(step_line[-1])

    # use original / factor:
    new_step_line = step_line[:-1] + [ '%.4E' % (original_increment/factor) ] 
    new_step_line_str = str(new_step_line[0])
    for i in range(1, len(new_step_line)):
        new_step_line_str = new_step_line_str + ', '
        new_step_line_str = new_step_line_str + str(new_step_line[i])
    new_step_line_str = new_step_line_str + '\n'
    with open(filename, 'w') as f:
        f.writelines(lines[:step_line_ind])
        f.writelines(new_step_line_str)
        f.writelines(lines[step_line_ind+1:])
    job_run()
    if check_complete():
        write_original(filename)
        return
    elif ct >= uset.recursion_depth:
        write_original(filename)
        return
    else:
        refine_run(ct)

def combine_SS(zeros):
    """
    Reads npy stress-strain output and appends current results.
    """
    filename = 'out_time_disp_force.npy'
    sheet = np.loadtxt( 'temp_time_disp_force.csv', delimiter=',', skiprows=1 ) 
    if zeros:
        sheet = np.zeros( (np.shape(sheet)) )
    if os.path.isfile( filename ): 
        dat = np.load( filename )
        dat = np.dstack( (dat,sheet) )
    else:
        dat = sheet
    np.save( filename, dat )

def calc_error():
    """
    Calculates root mean squared error between experimental and calculated 
    stress-strain curves.  
    """
    global exp_SS_file
    simSS = np.loadtxt( 'temp_time_disp_force.csv', delimiter=',', skiprows=1 )[:,1:]
    # TODO get simulation dimensions at beginning of running this file, pass to this function
    simSS[:,0] = simSS[:,0]/uset.length  # disp to strain
    simSS[:,1] = simSS[:,1]/uset.area    # force to stress

    # load experimental data
    expSS = np.loadtxt( exp_SS_file, skiprows=1, delimiter=',' )

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

    # smooth out simulated SS -- note this isn't real smoothing but maybe it should be
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
    deviations = np.asarray( [ smoothedSS( x_error_eval_pts[i] ) - fineSS[i] \
        for i in range(len(fineSS)) ] )
    rmse = np.sqrt( np.sum( deviations**2 ) / len(fineSS) ) 

    return rmse

def write_parameters(next_params):
    """
    Going in order of `uset.Param_list`, write new parameter values to the Abaqus 
    material input file. 
    """
    filename = [ f for f in os.listdir(os.getcwd()) if f.startswith('Mat_BW')][0]
    with open( filename, 'r' ) as f1:
        lines = f1.readlines()
    with open( 'temp_mat_file.inp', 'w+' ) as f2:    
        for line in lines:
            skip = False
            for param in uset.param_list:
                if line.startswith(param):
                    f2.write( param + ' = ' + str(next_params[uset.param_list.index(param)]) + '\n')
                    skip = True
            if not skip:
                f2.write( line )

    os.remove( filename )
    os.rename( 'temp_mat_file.inp', filename )


class Get_Fd(object):
    """
    Requires Abaqus-specific libraries, must be called from Abaqus python.
    """
    
    def __init__(self,ResultFile, step):

        CurrentPath = os.getcwd()
        self.ResultFilePath = os.path.join(CurrentPath, ResultFile + '.odb')

        self.Time = []
        self.TopU2 = []
        self.TopRF2 = []
        
        # step = 'Loading'
        # step = step_gl
        instance = 'PART-1-1'
        TopRPset = 'RP-TOP'
        
        odb = openOdb(path=self.ResultFilePath, readOnly=True)
        steps = odb.steps[step]
        frames = odb.steps[step].frames
        numFrames = len(frames)
        TopRP = odb.rootAssembly.instances[instance].nodeSets[TopRPset] # if node set is in Part
        #TopNodes = odb.rootAssembly.nodeSets[NodeSetTop] # if the node set is in Assembly
        
        for x in range(numFrames):
            Frame = frames[x]
            # Record time
            Time1 = Frame.frameValue
            self.Time.append(Time1) # list append
            # Top RP results
            Displacement = Frame.fieldOutputs['U']
            ReactionForce = Frame.fieldOutputs['RF']
            TopU  = Displacement.getSubset(region=TopRP).values
            TopRf = ReactionForce.getSubset(region=TopRP).values
            self.TopU2  = self.TopU2  + map(lambda x:x.data[1], TopU)  # list combination
            self.TopRF2 = self.TopRF2 + map(lambda x:x.data[1], TopRf) # list combination
        
        odb.close()

def write2file():
    """
    Using Get_Fd object, read odb file and write time-displacement-force to csv file.
    """
    job = [f for f in os.listdir(os.getcwd()) if f.endswith('.odb')][0][:-4]
    with open('temp_time_disp_force.csv','w') as f:
        f.write('{0},{1},{2}\n'.format('Time','U2','RF2'))
        for step in uset.loading_steps:
            # global step_gl
            # step_gl = step
            Result_Fd = Get_Fd(job, step)
            for i in range(len(Result_Fd.Time)):
                f.write('%.5f,' % Result_Fd.Time[i])
                f.write('%.5f,' % Result_Fd.TopU2[i])
                f.write('%.5f\n' % Result_Fd.TopRF2[i])

if __name__ == '__main__':
    main()
