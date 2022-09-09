"""
Script to optimize CPFEM parameters using as the engine Abaqus and Huang's subroutine.
All user inputs should be in `opt_input.py` file.
"""
import os
import shutil
import subprocess
import numpy as np
from numpy.linalg import norm
import opt_input as uset  # user settings file

try:  # Abaqus-specific imports
    from odbAccess import *
    from abaqusConstants import *
    from odbMaterial import *
    from odbSection import *
except:  # optimizer-specific imports
    from skopt import Optimizer
    from scipy.interpolate import interp1d
    from scipy.optimize import curve_fit, root


def main():
    remove_out_files()
    global exp_data, in_opt, opt_out
    exp_data = ExpData(uset.orientations)
    in_opt = InOpt(uset.orientations, uset.param_list, uset.param_bounds)
    opt_out = OptOut()
    opt = instantiate_optimizer(in_opt, uset.n_initial_points)
    if uset.do_load_previous: 
        opt = load_opt(opt)
    load_subroutine()

    loop(opt, uset.loop_len)


def instantiate_optimizer(in_opt, uset):
    opt = Optimizer(
        dimensions = in_opt.bounds, 
        base_estimator = 'gp',
        n_initial_points = uset.n_initial_points,
        initial_point_generator = 'lhs',
        acq_func = 'EI',
        acq_func_kwargs = {'xi':1.0} # default is 0.01, higher values favor exploration
    )
    return opt


def loop(opt, loop_len):
    def single_loop(opt, i):
        global opt_progress  # global progress tracker, row:(i, params, error)
        next_params = [round_sig(param) for param in opt.ask()]  # get and round parameters to test
        while param_check(uset.param_list):  # True if Tau0 >= TauS
            # this tells opt that params are bad but does not record it elsewhere
            opt.tell(next_params, max_rmse(i))
            next_params = [round_sig(param) for param in opt.ask()] 
        else:
            write_params(uset.param_file, in_opt.material_params, next_params[0:in_opt.num_params_material])
            for orient in uset.orientations.keys():
                if in_opt.has_orient_opt[orient]:
                    orient_components = get_orient_info(next_params, orient)
                    write_params('mat_orient.inp', orient_components['names'], orient_components['values'])
                else:
                    shutil.copy(uset.orientations[orient]['inp'], 'mat_orient.inp')
                shutil.copy('{0}_{1}.inp'.format(uset.jobname, orient), '{0}.inp'.format(uset.jobname))
                
                job_run()
                if not check_complete(): # try decreasing max increment size
                    refine_run()
                if not check_complete(): # if it still fails, write max_rmse, go to next parameterset
                    write_maxRMSE(i, next_params, opt)
                    return
                else:
                    output_fname = 'temp_time_disp_force_{0}.csv'.format(orient)
                    if os.path.isfile(output_fname): 
                        os.remove(output_fname)
                    job_extract(orient)  # extract data to temp_time_disp_force.csv
                    if np.sum(np.loadtxt(output_fname, delimiter=',', skiprows=1)[:,1:2]) == 0:
                        write_maxRMSE(i, next_params, opt)
                        return

            # error value:
            rmse_list = []
            for orient in uset.orientations.keys():
                rmse_list.append(calc_error(exp_data.data[orient]['raw'], orient))
                combine_SS(zeros=False, orientation=orient)  # save stress-strain data
            rmse = np.mean(rmse_list)
            opt.tell(next_params, rmse)
            opt_progress = update_progress(i, next_params, rmse)
            write_opt_progress()
    
    get_first()
    for i in range(loop_len):
        single_loop(opt, i)




class OptOut():
    def __init__(self):
        self.iterations = []
        self.param_values = []
        self.error_values = []
        self.errors_per_orient = []

    def add_error(self, param_values, error):
        iter_num = len(self.iterations) + 1
        self.iterations.append(iter_num)
        self.param_values.append(param_values)
        self.error_values.append(error)
        temp_error_dict = {}
        for orient in orientations:
            temp_error_dict[orient] = errors[orient]

# functions to be added to OptOut class:
def write_error_to_file(error_list, orient_list):
    error_fname = 'out_errors.txt'
    if os.path.isfile(error_fname):
        with open(error_fname, 'a+') as f:
            f.write(error_list + [np.mean(error_list)])
    else:
        with open(error_fname, 'w+') as f:
            f.write('# errors for {} and mean error'.format(orient_list))


def write_opt_progress():
    global opt_progress
    opt_progress_header = ','.join( ['iteration'] + in_opt.params + ['RMSE'])
    np.savetxt('out_progress.txt', opt_progress, delimiter='\t', header=opt_progress_header)


def update_progress(i, next_params, rmse):
    global opt_progress
    if (i == 0) and (uset.do_load_previous == False): 
        opt_progress = np.transpose(np.asarray([i] + next_params + [rmse]))
    else: 
        opt_progress = np.vstack((opt_progress, np.asarray([i] + next_params + [rmse])))
    return opt_progress


def write_maxRMSE(i, next_params, opt):
    global opt_progress
    rmse = max_rmse(i)
    opt.tell( next_params, rmse )
    for orientation in uset.orientations.keys():
        combine_SS(zeros=True, orientation=orientation)
    opt_progress = update_progress(i, next_params, rmse)
    write_opt_progress()


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
            np.where(opt_progress[:grace,-1] == uset.large_error))
        if len(errors) < np.round( grace/2 ):
            return uset.large_error
        else:
            iq1, iq3 = np.quantile(errors, [0.25,0.75])
            return np.ceil(np.mean(errors) + (iq3-iq1)*1.5)

def combine_SS(zeros, orientation):
    """
    Reads npy stress-strain output and appends current results.
    """
    filename = 'out_time_disp_force_{0}.npy'.format(orientation)
    sheet = np.loadtxt( 'temp_time_disp_force_{0}.csv'.format(orientation), delimiter=',', skiprows=1 )
    if zeros:
        sheet = np.zeros((np.shape(sheet)))
    if os.path.isfile(filename): 
        dat = np.load(filename)
        dat = np.dstack((dat,sheet))
    else:
        dat = sheet
    np.save(filename, dat)


def calc_error(exp_data, orientation):
    """
    Calculates root mean squared error between experimental and calculated 
    stress-strain curves.  
    """
    simSS = np.loadtxt('temp_time_disp_force_{0}.csv'.format(orientation), delimiter=',', skiprows=1)[1:,1:]
    # TODO get simulation dimensions at beginning of running this file, pass to this function
    simSS[:,0] = simSS[:,0] / uset.length  # disp to strain
    simSS[:,1] = simSS[:,1] / uset.area    # force to stress

    expSS = exp_data

    # deal with unequal data lengths 
    if simSS[-1,0] > expSS[-1,0]:
        # chop off simSS
        cutoff = np.where(simSS[:,0] > expSS[-1,0])[0][0] - 1
        simSS = simSS[:cutoff,:]
        cutoff_strain = simSS[-1,0]
    elif simSS[-1,0] < expSS[-1,0]:
        # chop off expSS
        cutoff = np.where(simSS[-1,0] < expSS[:,0])[0][0] - 1
        expSS = expSS[:cutoff,:]
        cutoff_strain = expSS[-1,0]
    else:
        cutoff_strain = simSS[-1,0]
    begin_strain = max(min(expSS[:,0]), min(simSS[:,0]))

    def powerlaw(x,k,n):
        y = k * x**n
        return y

    def fit_powerlaw(x,y):
        popt, _ = curve_fit(powerlaw,x,y)
        return popt

    # interpolate points in both curves
    num_error_eval_pts = 1000
    x_error_eval_pts = np.linspace(begin_strain, cutoff_strain, num = num_error_eval_pts)
    smoothedSS = interp1d(simSS[:,0], simSS[:,1])
    if not uset.i_powerlaw:
        smoothedExp = interp1d(expSS[:,0], expSS[:,1])
        fineSS = smoothedExp(x_error_eval_pts)
    else:
        popt = fit_powerlaw(expSS[:,0], expSS[:,1])
        fineSS = powerlaw(x_error_eval_pts, *popt)

    # strictly limit to interpolation
    while x_error_eval_pts[-1] >= expSS[-1,0]:
        fineSS = np.delete(fineSS, -1)
        x_error_eval_pts = np.delete(x_error_eval_pts, -1)

    # error function
    # for dual opt, error is normalized by exp value (root mean percent error instead of RMSE)
    deviations_pct = np.asarray([100*(smoothedSS(x_error_eval_pts[i]) - fineSS[i])/fineSS[i] \
        for i in range(len(fineSS))])
    rmse = np.sqrt(np.sum( deviations_pct**2) / len(fineSS)) 

    return rmse


class ExpData():
    def __init__(self, orientations):
        self.data = {}
        for orient in orientations.keys():
            expname = orientations[orient]['exp']
            # orientname = orientations[orient]['inp']
            jobname = uset.jobname + '_{0}.inp'.format(orient)
            self._max_strain = self._get_max_strain(expname)
            self.raw = self._get_SS(expname)
            self._write_strain_inp(jobname)
            self.data[orient] = {
                'max_strain':self._max_strain,
                'raw':self.raw
            }

    def _load(self, fname):
        """Load original exp_SS data, order it."""
        original_SS = np.loadtxt(fname, skiprows=1, delimiter=',')
        original_SS = original_SS[original_SS[:,0].argsort()]
        return original_SS

    def _get_max_strain(self, fname):
        """Take either user max strain or file max strain."""
        if float(uset.max_strain) == 0.0:
            max_strain = max(np.loadtxt(fname, skiprows=1, delimiter=',' )[:,0])
        else:
            max_strain = uset.max_strain
        return max_strain

    def _get_SS(self, fname):
        """Limit experimental data to within max_strain"""
        expSS = self._load(fname)
        max_strain = self._max_strain
        if not (float(uset.max_strain) == 0.0):
            max_point = 0
            while expSS[max_point,0] <= max_strain:
                max_point += 1
            expSS = expSS[:max_point, :]
        np.savetxt('temp_expSS.csv', expSS, delimiter=',')
        return expSS

    def _write_strain_inp(self, jobname):
        """Modify displacement B.C. in main Abaqus input file to match max strain."""
        # input file:
        max_bound = round(self._max_strain * uset.length, 4) #round to 4 digits

        with open('{0}.inp'.format(uset.jobname), 'r') as f:
            lines = f.readlines()

        # find last number after RP-TOP under *Boundary
        bound_line_ind = [ i for i, line in enumerate(lines) \
            if line.lower().startswith('*boundary')][0]
        bound_line_ind += [ i for i, line in enumerate(lines[bound_line_ind:]) \
            if line.strip().lower().startswith('rp-top')][0]
        bound_line = [number.strip() for number in lines[bound_line_ind].strip().split(',')]

        new_bound_line = bound_line[:-1] + [max_bound]
        new_bound_line_str = str(new_bound_line[0])

        for i in range(1, len(new_bound_line)):
            new_bound_line_str = new_bound_line_str + ', '
            new_bound_line_str = new_bound_line_str + str(new_bound_line[i])
        new_bound_line_str = '   ' + new_bound_line_str + '\n'

        # write to uset.jobname file
        with open(jobname, 'w') as f:
            f.writelines(lines[:bound_line_ind])
            f.writelines(new_bound_line_str)
            f.writelines(lines[bound_line_ind+1:])


def unit_vector(vector):
    return vector/norm(vector)

def get_orient_info(next_params, orient):
    """
    Get components of orientation-defining vectors and their names
    for substitution into the orientation input files.
    """
    dir_load = uset.orientations[orient]['offset']['dir_load']
    dir_0deg = uset.orientations[orient]['offset']['dir_0deg']

    if (orient+'_mag' in in_opt.params):
        index_mag = in_opt.params.index(orient+'_mag')
        angle_mag = next_params[index_mag]
    else:
        angle_mag = in_opt.fixed_vars[orient+'_mag']
    
    if (orient+'_deg' in in_opt.params):
        index_deg = in_opt.params.index(orient+'_deg')
        angle_deg = next_params[index_deg]
    else:
        angle_deg = in_opt.fixed_vars[orient+'_deg']

    col_load = unit_vector(np.asarray(dir_load))
    col_0deg = unit_vector(np.asarray(dir_0deg))
    col_cross = unit_vector(np.cross(col_load, col_0deg))

    basis_og = np.stack((col_load, col_0deg, col_cross), axis=1)
    basis_new = np.matmul(basis_og, _mk_x_rot(np.deg2rad(angle_deg)))
    dir_to = basis_new[:,1]

    if __debug__:  # write angle_deg rotation info
        dir_load = dir_load / norm(dir_load)
        dir_to = dir_to / norm(dir_to)
        dir_0deg = dir_0deg / norm(dir_0deg)
        with open('debug.txt', 'a+') as f:
            f.write('orientation: {}'.format(orient))
            f.write('\nbasis OG: \n{0}'.format(basis_og))
            f.write('\n')
            f.write('\nrotation: \n{0}'.format(_mk_x_rot(angle_deg*np.pi/180.)))
            f.write('\n')
            f.write('\nbasis new: \n{0}'.format(basis_new))
            f.write('\n\n')
            f.write('dir_load: {0}\tdir_to: {1}\n'.format(dir_load, dir_to))
            f.write('angle_deg_inp: {0}\n'.format(angle_deg))
            f.write('all params: {}'.format(next_params))

    sol = get_offset_angle(dir_load, dir_to, angle_mag)
    dir_tot = dir_load + sol * dir_to
    dir_ortho = np.array([1, 0, -dir_tot[0]/dir_tot[2]])

    if __debug__: # write final loading orientation info
        angle_output = np.arccos(np.dot(dir_tot, dir_load)/(norm(dir_tot)*norm(dir_load)))*180./np.pi
        with open('debug.txt', 'a+') as f:
            f.write('\ndir_tot: {0}'.format(dir_tot))
            f.write('\ndir_ortho: {0}'.format(dir_ortho))
            f.write('\nangle_mag_input: {}\tangle_mag_output: {}'.format(angle_mag, angle_output))
            f.write('\n\n')
    component_names = ['x1', 'y1', 'z1', 'u1', 'v1', 'w1']
    component_values = list(dir_ortho) + list(dir_tot)

    return {'names':component_names, 'values':component_values}


def _mk_x_rot(theta):
    """
    Generates rotation matrix for theta (radians) clockwise rotation 
    about first column of 3D basis when applied from right.
    """
    rot = np.array([[1,             0,              0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta),  np.cos(theta)]])
    return rot


def get_offset_angle(direction_og, direction_to, angle):
    def _opt_angle(offset_amt, direction_og, direction_to, angle):
        """
        Angle difference between original vector and new vector, which is
        made by small offset toward new direction.  Returns zero when offset_amt 
        produces new vector at desired angle.  Uses higher namespace variables so 
        that the single argument can be tweaked by optimizer.
        """
        direction_new = direction_og + offset_amt * direction_to
        angle_difference = \
            np.dot(direction_og, direction_new) / \
            ( norm(direction_og) * norm(direction_new) ) \
            - np.cos(np.deg2rad(angle))
        return angle_difference

    sol = root(_opt_angle, 0.01, args=(direction_og, direction_to, angle), tol=1e-10).x
    return sol


class InOpt:
    def __init__(self, orientations, param_list, param_bounds):
        """Sorted orientations here defines order for use in single list passed to optimizer."""
        self.orients = sorted(orientations.keys())
        self.params, self.bounds, \
        self.material_params, self.material_bounds, \
        self.orient_params, self.orient_bounds \
            = ([] for i in range(6))
        for i in range(len(param_list)):
            self.material_params.append(param_list[i])
            self.material_bounds.append(param_bounds[i])
        
        # add orientation offset info:
        self.offsets = []
        self.has_orient_opt = {}
        self.fixed_vars = {}
        for orient in self.orients:
            if 'offset' in orientations[orient].keys():
                self.has_orient_opt[orient] = True
                self.offsets.append({orient:orientations[orient]['offset']})
                # ^ saves all info (TODO: check if still needed)
                
                # deg rotation *about* loading orientation:
                if isinstance(orientations[orient]['offset']['deg_bounds'], tuple):
                    self.orient_params.append(orient+'_deg')
                    self.orient_bounds.append(orientations[orient]['offset']['deg_bounds'])
                else:
                    self.fixed_vars[(orient+'_deg')] = orientations[orient]['offset']['deg_bounds']
                
                # mag rotation *away from* loading:
                if isinstance(orientations[orient]['offset']['mag_bounds'], tuple):
                    self.orient_params.append(orient+'_mag')
                    self.orient_bounds.append(orientations[orient]['offset']['mag_bounds'])
                else:
                    self.fixed_vars[(orient+'_mag')] = orientations[orient]['offset']['mag_bounds']
                
            else:
                self.has_orient_opt[orient] = False
        
        # combine material and orient info into one ordered list:
        self.params = self.material_params + self.orient_params
        self.bounds = as_float_tuples(self.material_bounds + self.orient_bounds)
        
        # descriptive stats on input object:
        self.num_params_material = len(self.material_params)
        self.num_params_orient = len(self.orient_params)
        self.num_params_total = len(self.params)


def as_float_tuples(list_of_tuples):
    """
    Take list of tuples that may include ints and return list of tuples containing only floats.
    Useful for optimizer param bounds since type of input determines type of param guesses.
    """
    new_list = []
    for tup in list_of_tuples:
        float_tup = tuple(float(value) for value in tup)
        new_list.append(float_tup)
    return new_list


def round_sig(x, sig=4):
    if x == 0.0: return 0.
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)


def load_subroutine():
    """Compile the user subroutine uset.umat as a shared library in the directory"""
    subprocess.run('abaqus make library=' + uset.umat, shell=True)


def job_run():
    subprocess.run( 
        'abaqus job=' + uset.jobname \
        + ' user=' + uset.umat[:-2] + '-std.o' \
        + ' cpus=' + str(uset.cpus) \
        + ' double int ask_delete=OFF', shell=True
        )


def job_extract(outname):
    run_string = 'abaqus python -c "from opt_fea import write2file; write2file()"'
    subprocess.run(run_string, shell=True)
    os.rename('temp_time_disp_force.csv', 'temp_time_disp_force_{0}.csv'.format(outname))


def get_first():
    """
    Run one simulation, using its output dimensions to get shape of output data.
    """
    job_run()
    have_1st = check_complete()
    if not have_1st: 
        refine_run()
    job_extract('initial')


def load_opt(opt):
    """
    Load input files of previous optimizations to use as initial points in current optimization.
    Note the parameter bounds for the input files must be within current parameter bounds.
    """
    global opt_progress
    filename = 'out_progress.txt'
    arrayname = 'out_time_disp_force.npy'
    opt_progress = np.loadtxt(filename, skiprows=1)
    # renumber iterations (negative length to zero) to distinguish from new calculations:
    opt_progress[:,0] = np.array([i for i in range(-1*len(opt_progress[:,0]),0)])
    x_in = opt_progress[:,1:-1].tolist()
    y_in = opt_progress[:,-1].tolist()

    if __debug__:
        with open('debug.txt', 'a+') as f:
            f.write('loading previous results\n')
            f.writelines(['x_in: {0}\ty_in: {1}'.format(x,y) for x,y in zip(x_in, y_in)])

    opt.tell(x_in, y_in)
    return opt


def remove_out_files():
    if not uset.do_load_previous:
        out_files = [f for f in os.listdir(os.getcwd()) \
            if (f.startswith('out_') or f.startswith('res_') or f.startswith('temp_'))]
        if len(out_files) > 0:
            for f in out_files:
                os.remove(f)
    job_files = [f for f in os.listdir(os.getcwd()) \
        if (f.startswith(uset.jobname)) and not (f.endswith('.inp'))]


def param_check(param_list):
    """
    True if tau0 >= tauS, which is bad, not practically but in theory.
    In theory, tau0 should always come before tauS.
    """
    # TODO: ck if it's possible to satisfy this based on mat_params and bounds, raise helpful error
    tau0_list, tauS_list = [], []
    for sysnum in ['', '1', '2']:
        if ('TauS'+sysnum in param_list) or ('Tau0'+sysnum in param_list):
            f1 = open(uset.param_file, 'r')
            lines = f1.readlines()
            for line in lines:
                if line.startswith('Tau0'+sysnum): tau0_list.append(float(line[7:]))
                if line.startswith('TauS'+sysnum): tauS_list.append(float(line[7:]))
            f1.close()
    is_bad = any([(tau0 >= tauS) for tau0, tauS in zip(tau0_list, tauS_list)])
    return is_bad


def check_complete():
    """
    Return true if Abaqus has finished sucessfully.
    """
    stafile = uset.jobname + '.sta'
    if os.path.isfile(stafile):
            last_line = str(subprocess.check_output(['tail', '-1', stafile]))
    else: 
        last_line = ''
    return ('SUCCESSFULLY' in last_line)


def refine_run(ct=0):
    """
    Cut max increment size by `factor`, possibly multiple times (up to 
    `uset.recursion_depth` or until Abaqus finished successfully).
    """
    factor = 5.0
    ct += 1
    # remove old lock file from previous unfinished simulation
    subprocess.run('rm *.lck', shell=True)
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
        if line.lower().startswith('*static')][0] + 1 
    step_line = [number.strip() for number in lines[step_line_ind].strip().split(',')]
    original_increment = float(step_line[-1])

    # use original / factor:
    new_step_line = step_line[:-1] + ['%.4E' % (original_increment/factor)]
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


def write_params(fname, param_names, param_values):
    """
    Write parameter values to file with `=` as separator.
    Used for material and orientation input files.
    'param_names': list of strings, shares order with 'param_values'
    """
    with open(fname, 'r') as f1:
        lines = f1.readlines()
    with open('temp_' + fname, 'w+') as f2:
        for line in lines:
            skip = False
            for param_name, param_value in zip(param_names, param_values):
                if line[:line.find('=')].strip() == param_name:
                    f2.write(param_name + ' = ' + str(param_value) + '\n')
                    skip = True
            if not skip:
                f2.write(line)
    os.remove(fname)
    os.rename('temp_' + fname, fname)


class GetForceDisplacement(object):
    """
    Requires Abaqus-specific libraries, must be called from Abaqus python.
    """
    
    def __init__(self,ResultFile):

        CurrentPath = os.getcwd()
        self.ResultFilePath = os.path.join(CurrentPath, ResultFile + '.odb')

        self.Time = []
        self.TopU2 = []
        self.TopRF2 = []
        
        step = 'Loading'
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
    Using GetForceDisplacement object, read odb file and write time-displacement-force to csv file.
    """
    job = [f for f in os.listdir(os.getcwd()) if f.endswith('.odb')][0][:-4]
    Result_Fd = GetForceDisplacement(job)
    with open('temp_time_disp_force.csv','w') as f:
        f.write('{0},{1},{2}\n'.format('Time','U2','RF2'))
        for i in range(len(Result_Fd.Time)):
            f.write('%.5f,' % Result_Fd.Time[i])
            f.write('%.5f,' % Result_Fd.TopU2[i])
            f.write('%.5f\n' % Result_Fd.TopRF2[i])


if __name__ == '__main__':
    main()
