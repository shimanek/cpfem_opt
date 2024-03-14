import numpy as np
from copy import deepcopy
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import warnings

from matmdl.core.parser import uset
# from matmdl.optimizer import update_progress


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
    interpSim = interp1d(simSS[:,0], simSS[:,1])
    if not uset.i_powerlaw:
        interpExp = interp1d(expSS[:,0], expSS[:,1])
    else:
        popt = fit_powerlaw(expSS[:,0], expSS[:,1])
        def interpExp(x):
            return powerlaw(x, *popt)

    # strictly limit to interpolation
    while x_error_eval_pts[-1] >= expSS[-1,0]:
        x_error_eval_pts = np.delete(x_error_eval_pts, -1)

    # error function
    stress_error = _stress_diff(x_error_eval_pts, interpSim, interpExp)
    slope_error = _slope_diff(x_error_eval_pts, interpSim, interpExp)
    w = uset.slope_weight
    error = (1-w)*stress_error + w*slope_error
    # print(f"DBG:err: stress, slope, mean: {stress_error}, {slope_error}, {error}")
    return error


def _stress_diff(x, curve1, curve2):
    """ 
    root mean percent error between curves

    Args:
        x: array of values at which to evaluate curve differences
        curve1: f(x) for test curve
        curve2: f(x) for reference curve
    """
    ycurve1 = curve1(x)
    ycurve2 = curve2(x)
    stress_error = ycurve1 - ycurve2

    norm_factor = np.mean(np.abs(ycurve2))
    error = np.sqrt(np.sum(stress_error**2) / len(stress_error)) / norm_factor * 100
    return error


def ddx_pointwise(curve, x):
    """Give point-to-point slope values of curve over x"""
    return (curve(x[1:]) - curve(x[:-1])) / (x[1:] - x[:-1])


def ddx_rolling(curve, x, window):
    """Give rolling window slope of curve"""
    n = int(window)
    assert n > 1, "Rolling average requires window width of 2 or more points"
    num_windows = len(x) - window
    slopes = np.empty(num_windows)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        for i in range(num_windows):
            slopes[i] = np.polyfit(x[i:i+n], curve(x[i:i+n]), 1)[0]
    return slopes


def _slope_diff(x, curve1, curve2):
    """ 
    estimate of slope error between curves

    Args:
        x: array of values at which to evaluate slope differences
        curve1: f(x) for test curve
        curve2: f(x) for reference curve

    Returns:
        error: summed percent differences in slopes
    """
    window_width = 15
    # x has 1k points, exp usually has 200 and window of 3 works ok, so use 15
    dcurve1 = ddx_rolling(curve1, x, window_width)
    dcurve2 = ddx_rolling(curve2, x, window_width)
    slope_diffs = dcurve1 - dcurve2

    norm_factor = np.mean(np.abs(dcurve2))

    error = np.sqrt(np.sum(slope_diffs**2) / (len(slope_diffs))) / norm_factor * 100
    return error
