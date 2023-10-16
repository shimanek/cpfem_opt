import os
import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from matmdl.parser import uset
from matmdl.runner import combine_SS
from matmdl.optimizer import update_progress, write_opt_progress


def calc_error(
        exp_data: 'Nx2 matrix', 
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



def write_maxRMSE(i: int, next_params: tuple, opt: object, in_opt: object):
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
    rmse = max_rmse(i, opt_progress)
    opt.tell( next_params, rmse )
    for orientation in uset.orientations.keys():
        combine_SS(zeros=True, orientation=orientation)
    opt_progress = update_progress(i, next_params, rmse)
    write_opt_progress(in_opt)


def max_rmse(loop_number: int, opt_progress):
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
    if loop_number < grace:  # use user specified error
        return uset.large_error
    elif loop_number >= grace:  # use previous errors
        # first remove instances of user-specified error
        large_error_locs = np.where(opt_progress[:grace,-1] == uset.large_error)
        if len(large_error_locs) == 0:
            errors = opt_progress[:,-1]
        else:
            errors = np.concatenate(
                (
                    np.delete(opt_progress[:grace,-1], large_error_locs),
                    opt_progress[grace:,-1],
                ),
                axis=0
            )
        if len(errors) < np.round(grace/2):  # not enough error data
            return uset.large_error
        else:
            iq1, iq3 = np.quantile(errors, [0.25,0.75])
            return np.ceil(np.mean(errors) + (iq3-iq1)*1.5)
