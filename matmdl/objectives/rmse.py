import os
import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from matmdl.parser import uset
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
        # fineSS = smoothedExp(x_error_eval_pts)
        def fineSS(x):
            return smoothedExp(x)
    else:
        popt = fit_powerlaw(expSS[:,0], expSS[:,1])
        # fineSS = powerlaw(x_error_eval_pts, *popt)
        def fineSS(x):
            return powerlaw(x, *popt)

    # strictly limit to interpolation
    while x_error_eval_pts[-1] >= expSS[-1,0]:
        # fineSS = np.delete(fineSS, -1)
        x_error_eval_pts = np.delete(x_error_eval_pts, -1)

    # error function
    stress_error = _stress_diff(x_error_eval_pts, smoothedSS, fineSS)
    slope_error = _slope_diff(x_error_eval_pts, smoothedSS, fineSS)
    if hasattr(uset, "slope_weight"):
        w = uset.slope_weight
    else:
        w = 0.4
        print(f"warning, using default slope weight of {w}")
    error = (1-w)*stress_error + w*slope_error
    return error


def _stress_diff(x, curve1, curve2):
    """ 
    root mean percent error between curves

    Args:
        x: array of values at which to evaluate curve differences
        curve1: f(x) for test curve
        curve2: f(x) for reference curve
    """
    percent_error = (curve1(x) - curve2(x))/curve2(x)*100

    error = np.sqrt(np.sum(percent_error**2) / len(x))
    return error


def _slope_diff(x, curve1, curve2):
    """ 
    estimate of slope error between curves

    Args:
        x: array of values at which to evaluate slope differences
        curve1: f(x) for test curve
        curve2: f(x) for reference curve
    """
    def ddx(curve, x):
        return (curve(x[1:]) - curve(x[:-1])) / (x[1:] - x[:-1])

    dcurve1 = ddx(curve1, x)
    dcurve2 = ddx(curve2, x)
    slope_diffs = (dcurve1 - dcurve2) / (dcurve2) * 100

    error = np.sqrt(np.sum(slope_diffs**2) / (len(x) - 1))
    return error


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
