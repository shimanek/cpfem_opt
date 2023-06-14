"""
Script to optimize CPFEM parameters using as the engine Abaqus and Huang's subroutine.
All user inputs should be in the ``opt_input.py`` file.
Two import modes: tries to load Abaqus modules (when running to extract stress-strain
info) or else imports sciki-optimize library (when running as main outside of Abaqus).
"""
import os
import shutil
import subprocess
import numpy as np
from numpy.linalg import norm
from copy import deepcopy
import opt_input as uset  # user settings file

from skopt import Optimizer
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, root

from typing import Union
from nptyping import NDArray, Shape, Floating

from matmdl.components.utilities import unit_vector, as_float_tuples, round_sig


def main():
    """Instantiate data structures, start optimization loop."""
    remove_out_files()
    global exp_data, in_opt
    # TODO declare out_progress global up here?
    exp_data = ExpData(uset.orientations)
    in_opt = InOpt(uset.orientations, uset.params)
    opt = instantiate_optimizer(in_opt, uset)
    if uset.do_load_previous: opt = load_opt(opt)
    load_subroutine()

    loop(opt, uset.loop_len)


def instantiate_optimizer(in_opt: object, uset: object) -> object:
    """
    Define all optimization settings, return optimizer object.

    Args:
        in_opt: Input settings defined in :class:`InOpt`.
        uset : User settings from input file.

    Returns:
        skopt.Optimize: Instantiated optimization object.
    """
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
    """Holds all optimization iteration instructions."""
    def single_loop(opt, i):
        """
        Run single iteration (one parameter set) of the optimization scheme.

        Single loops need to be separate function calls to allow empty returns to exit one
        parameter set.
        """
        global opt_progress  # global progress tracker, row:(i, params, error)
        next_params = get_next_param_set(opt, in_opt)
        write_params(uset.param_file, in_opt.material_params, next_params[0:in_opt.num_params_material])
        while param_check(uset.params):  # True if Tau0 >= TauS
            # this tells opt that params are bad but does not record it elsewhere
            opt.tell(next_params, max_rmse(i))
            next_params = get_next_param_set(opt, in_opt)
            write_params(uset.param_file, in_opt.material_params, next_params[0:in_opt.num_params_material])
        else:
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
            write_error_to_file(rmse_list, in_opt.orients)
            rmse = np.mean(rmse_list)
            opt.tell(next_params, rmse)
            opt_progress = update_progress(i, next_params, rmse)
            write_opt_progress()
    
    get_first(opt, in_opt)
    for i in range(loop_len):
        single_loop(opt, i)


def write_error_to_file(error_list: list[float], orient_list: list[str]) -> None:
    """
    Write error values separated by orientation, if applicable.

    Args:
        error_list: List of floats indicated error values for each orientation
            in ``orient_list``, with which this list shares an order.
        orient_list: List of strings holding orientation nicknames.
    """
    error_fname = 'out_errors.txt'
    if os.path.isfile(error_fname):
        with open(error_fname, 'a+') as f:
            f.write('\n' + ','.join([str(err) for err in error_list + [np.mean(error_list)]]))
    else:
        with open(error_fname, 'w+') as f:
            f.write('# errors for {} and mean error'.format(orient_list))


class ExpData():
    """
    Loads and stores experimental data.

    Attributes:
        data (dict): Indexed by orientation name defined in :ref:`orientations`, 
            with values of max strain (internal: ``_max_strain``) and ``raw``, 
            which houses the experimental stress strain data truncated by max strain.

    Note:
        Experimental stress-strain data are expected as plaintext in two columns:
        strain (unitless), and stress (matching the CPFEM inputs, often MPa).

    """
    def __init__(self, orientations: dict):
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

    def _load(self, fname: str):
        """
        Load original experimental stress-strain data and order it by strain.

        Args:
            fname: Filename for experimental stress-strain data
        """
        original_SS = np.loadtxt(fname, skiprows=1, delimiter=',')
        order = -1 if uset.is_compression else 1
        original_SS = original_SS[original_SS[:,0].argsort()][::order]
        return original_SS

    def _get_max_strain(self, fname: str):
        """
        Take either user max strain or file max strain.
        
        Args:
            fname: Filename for experimental stress-strain data
        """
        if float(uset.max_strain) == 0.0:
            if uset.is_compression == True:
                max_strain = min(np.loadtxt(fname, skiprows=1, delimiter=',' )[:,0])
            else:
                max_strain = max(np.loadtxt(fname, skiprows=1, delimiter=',' )[:,0])
        else:
            max_strain = uset.max_strain if not uset.is_compression else (-1 * uset.max_strain)
        return max_strain

    def _get_SS(self, fname: str):
        """
        Limit experimental data to within max_strain. 
        
        Args:
            fname: Filename for experimental stress-strain data
        """
        expSS = self._load(fname)
        max_strain = self._max_strain
        if not (float(uset.max_strain) == 0.0):
            max_point = 0
            while expSS[max_point,0] <= max_strain:
                max_point += 1
            expSS = expSS[:max_point, :]
        np.savetxt('temp_expSS.csv', expSS, delimiter=',')
        return expSS

    def _write_strain_inp(self, jobname: str):
        """
        Modify boundary conditions in main Abaqus input file to match max strain.
        
        Args:
            jobname: Filename for main Abaqus job -- unique to 
                orientation if applicable.

        Note:
            Relies on finding ``RP-TOP`` under ``*Boundary`` keyword in main
            input file.
        """
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


def get_orient_info(next_params: list, orient: str) -> dict:
    """
    Get components of orientation-defining vectors and their names
    for substitution into the orientation input files.

    Args:
        next_params: Next set of parameters to be evaluated
            by the optimization scheme.
        orient: Index string for dictionary of input
            orientations specified in :ref:`orientations`.
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
        with open('out_debug.txt', 'a+') as f:
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
        with open('out_debug.txt', 'a+') as f:
            f.write('\ndir_tot: {0}'.format(dir_tot))
            f.write('\ndir_ortho: {0}'.format(dir_ortho))
            f.write('\nangle_mag_input: {}\tangle_mag_output: {}'.format(angle_mag, angle_output))
            f.write('\n\n')
    component_names = ['x1', 'y1', 'z1', 'u1', 'v1', 'w1']
    component_values = list(dir_ortho) + list(dir_tot)

    return {'names':component_names, 'values':component_values}


def _mk_x_rot(theta: float) -> NDArray[Shape['3,3'], Floating]:
    """
    Generates rotation matrix for theta (radians) clockwise rotation 
    about first column of 3D basis when applied from right.
    """
    rot = np.array([[1,             0,              0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta),  np.cos(theta)]])
    return rot


def get_offset_angle(
    direction_og: NDArray[Shape['3'], Floating], 
    direction_to: NDArray[Shape['3'], Floating], 
    angle: float) -> object:
    """
    Iterative solution to finding vectors tilted toward other vectors.

    Args:
        direction_og: Real space vector defining
            the original direction to be tilted away from.
        direction_to: Real space vector defining
            the direction to be tilted towards.
        angle: The angle, in degrees, by which to tilt.

    Returns:
        scipy.optimize.OptimizeResult:
            A scipy object containing the attribute
            ``x``, the solution array, which, in this case, is a scalar
            multiplier such that the angle between ``direction_og``
            and ``sol.x`` * ``direction_to`` is ``angle``.

    """
    def _opt_angle(
        offset_amt: float, 
        direction_og: NDArray[Shape['3'], Floating], 
        direction_to: NDArray[Shape['3'], Floating], 
        angle: float):
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
    """
    Stores information about the optimization input parameters.

    Since the hardening parameters and orientation parameters are fundamentally
    different, this object stores information about both in such a way that they
    can still be access independently.

    Args:
        orientations (dict): Orientation information directly from ``opt_input``.
        param_list (list): List of material parameters to be optimized.
        param_bounds (list): List of tuples describing optimization bounds for
            variables given in ``param_list``.

    Attributes:
        orients (list): Nickname strings defining orientations, as given
            in :ref:`orientations`.
        material_params (list): Parameter names to be optimized, as in :ref:`orientations`.
        material_bounds (list): Tuple of floats defining bounds of parameter in the same
            index of ``self.params``, again given in :ref:`orientations`.
        orient_params (list): Holds orientation parameters to be optimized, or single
            orientation parameters if not given as a tuple in :ref:`orientations`.
            These are labeled ``orientationNickName_deg`` for the degree value of the
            right hand rotation about the loading axis and ``orientationNickName_mag``
            for the magnitude of the offset.
        orient_bounds (list): List of tuples corresponding to the bounds for the parameters
            stored in ``self.orient_params``.
        params (list): Combined list consisting of both ``self.material_params`` and
            ``self.orient_params``.
        bounds (list): Combined list consisting of both ``self.material_bounds`` and
            ``self.orient_bounds``.
        has_orient_opt (dict): Dictionary with orientation nickname as key and boolean
            as value indicating whether slight loading offsets should be considered
            for that orientation.
        fixed_vars (dict): Dictionary with orientation nickname as key and any fixed
            orientation information (``_deg`` or ``_mag``) for that loading orientation
            that is not going to be optimized.
        offsets (list): List of dictionaries containing all information about the offset
            as given in the input file. Not used/called anymore?
        num_params_material (int): Number of material parameters to be optimized.
        num_params_orient (int): Number of orientation parameters to be optimized.
        num_params_total (int): Number of parameters to be optimized in total.

    Note:
        Checks if ``orientations[orient]['offset']['deg_bounds']``
        in :ref:`orientations` is a tuple to determine whether
        orientation should also be optimized.
    """
    # TODO: check if ``offsets`` attribute is still needed.
    def __init__(self, orientations, params):
        """Sorted orientations here defines order for use in single list passed to optimizer."""
        self.orients = sorted(orientations.keys())
        self.params, self.bounds, \
        self.material_params, self.material_bounds, \
        self.orient_params, self.orient_bounds \
            = ([] for i in range(6))
        for param, bound in params.items():
            if type(bound) in (list, tuple):
                self.material_params.append(param)
                self.material_bounds.append(bound)
            elif type(bound) in (float, int):
                write_params(uset.param_file, param, float(bound))
            else:
                raise TypeError('Incorrect bound type in input file.')
        
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


def get_next_param_set(opt: object, in_opt: object) -> list[float]:
    """
    Give next parameter set to try using current optimizer state.

    Allow to sample bounds exactly, round all else to reasonable precision.
    """
    raw_params = opt.ask()
    new_params = []
    for param, bound in zip(raw_params, in_opt.bounds):
        if param in bound:
            new_params.append(param)
        else:
            new_params.append(round_sig(param, sig=6))
    return new_params


def load_subroutine():
    """
    Compile the user subroutine uset.umat as a shared library in the directory.
    """
    subprocess.run('abaqus make library=' + uset.umat, shell=True)


def write_opt_progress():
    """Writes global variable ``opt_progress`` to file."""
    global opt_progress
    opt_progress_header = ','.join( ['iteration'] + in_opt.params + ['RMSE'])
    np.savetxt('out_progress.txt', opt_progress, delimiter='\t', header=opt_progress_header)


def update_progress(i:int, next_params:tuple, error:float) -> None:
    """
    Writes parameters and error value to global variable ``opt_progress``.

    Args:
        i: Optimization iteration loop number.
        next_params: Parameter values evaluated during iteration ``i``.
        error: Error value of these parameters, which is defined in 
            :func:`calc_error`.
    """
    global opt_progress
    if (i == 0) and (uset.do_load_previous == False): 
        opt_progress = np.transpose(np.asarray([i] + next_params + [error]))
    else: 
        opt_progress = np.vstack((opt_progress, np.asarray([i] + next_params + [error])))
    return opt_progress


def write_maxRMSE(i: int, next_params: tuple, opt: object):
    """
    Write parameters and maximum error to global variable ``opt_progress``.

    Also tells the optimizer that this parameter set was bad. Error value
    determined by :func:`max_rmse`.

    Args:
        i : Optimization iteration loop number.
        next_params: Parameter values evaluated during iteration ``i``.
        opt: Current instance of skopt.Optimizer object.
    """
    global opt_progress
    rmse = max_rmse(i)
    opt.tell( next_params, rmse )
    for orientation in uset.orientations.keys():
        combine_SS(zeros=True, orientation=orientation)
    opt_progress = update_progress(i, next_params, rmse)
    write_opt_progress()


def job_run():
    """Run the Abaqus job!"""
    subprocess.run( 
        'abaqus job=' + uset.jobname \
        + ' user=' + uset.umat[:-2] + '-std.o' \
        + ' cpus=' + str(uset.cpus) \
        + ' double int ask_delete=OFF', shell=True
    )


def job_extract(outname: str):
    """
    Call :class:`GetForceDisplacement` from new shell to extract force-displacement data.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    extractions_script_path = os.path.join(src_dir, "opt_abaqus.py")
    run_string = f'abaqus python {extractions_script_path}'
    subprocess.run(run_string, shell=True)
    os.rename('temp_time_disp_force.csv', 'temp_time_disp_force_{0}.csv'.format(outname))


def get_first(opt: object, in_opt: object) -> None:
    """
    Run one simulation so its output dimensions can later inform the shape of output data.
    """
    job_run()
    if not check_complete():
        refine_run()
    job_extract('initial')


def load_opt(opt: object) -> object:
    """
    Load input files of previous optimizations to use as initial points in current optimization.
    
    Looks for a file named ``out_progress.txt`` from which to load previous results.
    Requires access to global variable ``opt_progress`` that stores optimization output. 
    The parameter bounds for the input files must be within current parameter bounds.
    Renumbers old/loaded results in ``opt_progress`` to have negative iteration numbers.

    Args:
        opt: Current instance of the optimizer object.

    Returns:
        skopt.Optimizer: Updated instance of the optimizer object.
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
        with open('out_debug.txt', 'a+') as f:
            f.write('loading previous results\n')
            f.writelines(['x_in: {0}\ty_in: {1}'.format(x,y) for x,y in zip(x_in, y_in)])

    opt.tell(x_in, y_in)
    return opt


def remove_out_files():
    """Delete files from previous optimization runs if not reloading results."""
    if not uset.do_load_previous:
        out_files = [f for f in os.listdir(os.getcwd()) \
            if (f.startswith('out_') or f.startswith('res_') or f.startswith('temp_'))]
        if len(out_files) > 0:
            for f in out_files:
                os.remove(f)
    job_files = [f for f in os.listdir(os.getcwd()) \
        if (f.startswith(uset.jobname)) and not (f.endswith('.inp'))]


def param_check(param_list: list[str]):
    """
    True if tau0 >= tauS

    In theory, tau0 should always come before tauS, even though it doesn't make a difference
    mathematically/practically. Function checks for multiple systems if numbered in the form
    ``TauS``, ``TauS1``, ``TauS2`` and ``Tau0``, ``Tau01``, ``Tau02``.
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


def max_rmse(loop_number: int):
    """
    Give a "large" error value.

    Return an estimate of a large enough error value to dissuade the optimizer 
    from repeating areas in parameter space where Abaqus+UMAT can't complete calculations.
    Often this ends up defaulting to `uset.large_error`, but a closer match to realistic
    errors is desireable so that the optimizer sees a smoother and more reaslisti function.

    Note:
        Grace period of 15 iterations is hardcoded here, as is the factor of 1.5 times the 
        interquartile range of previous error values.
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


def check_complete():
    """
    Return ``True`` if Abaqus has finished sucessfully.
    """
    stafile = uset.jobname + '.sta'
    if os.path.isfile(stafile):
            last_line = str(subprocess.check_output(['tail', '-1', stafile]))
    else: 
        last_line = ''
    return ('SUCCESSFULLY' in last_line)


def refine_run(ct: int=0):
    """
    Restart simulation with smaller maximum increment size.

    Cut max increment size by ``factor`` (hardcoded), possibly multiple 
    times up to ``uset.recursion_depth`` or until Abaqus finished successfully.
    After eventual success or failure, rewrites original input file so that the 
    next run starts with the initial, large maximum increment. 
    Recursive calls tracked through ``ct`` parameter.

    Args:
        ct: Number of times this function has already been called. Starts
        at 0 and can go up to ``uset.recursion_depth``.
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


def combine_SS(zeros: bool, orientation: str) -> None:
    """
    Reads npy stress-strain output and appends current results.

    Loads from ``temp_time_disp_force_{orientation}.csv`` and writes to 
    ``out_time_disp_force_{orientation}.npy``. Should only be called after all
    orientations have run, since ``zeros==True`` if any one fails.

    Args:
        zeros: True if the run failed and a sheet of zeros should be written
            in place of real time-force-displacement data.
        orientation: Orientation nickname to keep temporary output files separate.
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


def calc_error(
        exp_data: NDArray[Shape['*, 2'], Floating], 
        orientation: str
    ) -> float:
    """
    Give error value for run compared to experimental data.

    Calculates relative (%) root mean squared error between experimental and calculated 
    stress-strain curves. Interpolation of experimental data depends on :ref:`i_powerlaw`.

    Args:
        exp_data: Array of experimental strain-stress, as from 
            ``exp_data.data[orientation]['raw']``.
        orientation: Orientation nickname.
    """
    simSS = np.loadtxt('temp_time_disp_force_{0}.csv'.format(orientation), delimiter=',', skiprows=1)[1:,1:]
    # TODO get simulation dimensions at beginning of running this file, pass to this function
    simSS[:,0] = simSS[:,0] / uset.length  # disp to strain
    simSS[:,1] = simSS[:,1] / uset.area    # force to stress

    expSS = deepcopy(exp_data)

    if uset.is_compression:
        expSS *= -1.
        simSS *= -1.
    
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


def write_params(
        fname: str, 
        param_names: Union[list[str], str], 
        param_values: Union[list[float], float],
    ) -> None:
    """
    Write parameter values to file with ``=`` as separator.

    Used for material and orientation input files.

    Args:
        fname: Name of file in which to look for parameters.
        param_names: List of strings (or single string) describing parameter names.
            Shares order with ``param_values``.
        param_values: List of parameter values (or single value) to be written.
            Shares order with ``param_names``.
    """
    if ((type(param_names) not in (list, tuple)) or (len(param_names) == 1)) and (
        (type(param_values) not in (list, tuple)) or (len(param_values) == 1)
    ):
        param_names = [param_names]
        param_values = [param_values]
    elif len(param_names) != len(param_values):
        raise IndexError('Length of names must match length of values.')

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


if __name__ == '__main__':
    main()
