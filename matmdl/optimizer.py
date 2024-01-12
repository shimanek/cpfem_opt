from matmdl.runner import write_input_params
from matmdl.utilities import as_float_tuples, round_sig
from matmdl.parallel import Checkout
from skopt import Optimizer
from matmdl.parser import uset
from matmdl.state import state
import numpy as np
import os


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
            if type(bound) in (list, tuple):  # pass ranges to optimizer
                self.material_params.append(param)
                self.material_bounds.append([float(b) for b in bound])
            elif type(bound) in (float, int):  # write single values to file
                write_input_params(uset.param_file, param, float(bound))
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
                if isinstance(orientations[orient]['offset']['deg_bounds'], (tuple, list)):
                    self.orient_params.append(orient+'_deg')
                    self.orient_bounds.append(
                        [float(f) for f in orientations[orient]['offset']['deg_bounds']]
                    )
                else:
                    self.fixed_vars[(orient+'_deg')] = orientations[orient]['offset']['deg_bounds']

                # mag rotation *away from* loading:
                if isinstance(orientations[orient]['offset']['mag_bounds'], (tuple, list)):
                    self.orient_params.append(orient+'_mag')
                    self.orient_bounds.append(
                        [float(f) for f in orientations[orient]['offset']['mag_bounds']]
                    )
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


# def update_progress(i:int, next_params:tuple, error:float) -> None:
#     """
#     Writes parameters and error values to State.

#     Args:
#         i: Optimization iteration loop number.
#         next_params: Parameter values evaluated during iteration ``i``.
#         error: Error value of these parameters, which is defined in 
#             :func:`calc_error`.
#     """
#     global opt_progress
#     if (i == 0) and (uset.do_load_previous is False): 
#         opt_progress = np.transpose(np.asarray([i] + next_params + [error]))
#     else: 
#         opt_progress = np.vstack((opt_progress, np.asarray([i] + next_params + [error])))
#     return opt_progress


def load_opt(opt: object, search_local:bool=False) -> object:
    """
    Load input files of previous optimizations to use as initial points in current optimization.
    
    Looks for a file named ``out_progress.txt`` from which to load previous results.
    Requires access to global variable ``opt_progress`` that stores optimization output. 
    The parameter bounds for the input files must be within current parameter bounds.
    Renumbers old/loaded results in ``opt_progress`` to have negative iteration numbers.

    Args:
        opt: Current instance of the optimizer object.
        search_local: Look in the current directory for files 
            (convenient for plotting from parallel instances).

    Returns:
        skopt.Optimizer: Updated instance of the optimizer object.
    """
    global opt_progress
    filename = 'out_progress.txt'
    arrayname = 'out_time_disp_force.npy'
    if uset.main_path not in [os.getcwd(), "."] and not search_local:
        filename = os.path.join(uset.main_path, filename)
        arrayname = os.path.join(uset.main_path, arrayname)
    opt_progress = np.loadtxt(filename, skiprows=1, delimiter=',')
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
