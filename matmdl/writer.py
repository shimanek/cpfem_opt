"""
module for writing to files
"""
from matmdl.parser import uset
from matmdl.objectives.rmse import max_rmse
from matmdl.parallel import Checkout
from matmdl.optimizer import update_progress
import numpy as np
import os


def write_opt_progress(
        in_opt: object,
        opt_progress: object,
    ) -> None:
    """Appends last iteration infor of global variable ``opt_progress`` to file."""

    opt_progress_header = ['iteration'] + in_opt.params + ['RMSE']
    out_fpath = os.path.join(uset.main_path, 'out_progress.txt')

    if len(np.shape(opt_progress)) > 1:
        new_progress = opt_progress[-1,:]
    else:
        new_progress = opt_progress[:]

    add_header = not os.path.isfile(out_fpath)
    with open(out_fpath, "a+") as f:
        if add_header:
            header_padded = []
            for col_name in opt_progress_header:
                num_spaces = 8+6 - len(col_name)
                # 8 decimals, 6 other digits
                header_padded.append(col_name + num_spaces*" ")
            f.write(', '.join(header_padded) + "\n")
        f.write(',\t'.join([f"{a:.8e}" for a in new_progress]) + "\n")


def combine_SS(zeros: bool, orientation: str) -> None:
    """
    Reads npy stress-strain output and appends current results.

    Loads from ``temp_time_disp_force_{orientation}.csv`` and writes to 
    ``out_time_disp_force_{orientation}.npy``. Should only be called after all
    orientations have run, since ``zeros==True`` if any one fails.

    For parallel, needs to be called within a parallel.Checkout guard.

    Args:
        zeros: True if the run failed and a sheet of zeros should be written
            in place of real time-force-displacement data.
        orientation: Orientation nickname to keep temporary output files separate.
    """
    filename = os.path.join(uset.main_path, 'out_time_disp_force_{0}.npy'.format(orientation))
    sheet = np.loadtxt('temp_time_disp_force_{0}.csv'.format(orientation), delimiter=',', skiprows=1)
    if zeros:
        sheet = np.zeros((np.shape(sheet)))
    if os.path.isfile(filename): 
        dat = np.load(filename)
        dat = np.dstack((dat,sheet))
    else:
        dat = sheet
    np.save(filename, dat)
    # TODO: need to append???


def write_maxRMSE(i: int, next_params: tuple, opt: object, in_opt: object, opt_progress):
    """
    Write parameters and maximum error to global variable ``opt_progress``.

    Also tells the optimizer that this parameter set was bad. Error value
    determined by :func:`max_rmse`.

    Args:
        i : Optimization iteration loop number.
        next_params: Parameter values evaluated during iteration ``i``.
        opt: Current instance of skopt.Optimizer object.
    """
    rmse = max_rmse(i, opt_progress)
    opt.tell( next_params, rmse )
    for orientation in uset.orientations.keys():
        combine_SS(zeros=True, orientation=orientation)
    opt_progress = update_progress(i, next_params, rmse)
    write_opt_progress(in_opt)


def write_error_to_file(error_list: list[float], orient_list: list[str]) -> None:
    """
    Write error values separated by orientation, if applicable.

    Args:
        error_list: List of floats indicated error values for each orientation
            in ``orient_list``, with which this list shares an order.
        orient_list: List of strings holding orientation nicknames.
    """
    error_fpath = os.path.join(uset.main_path, 'out_errors.txt')
    if os.path.isfile(error_fpath):
        with open(error_fpath, 'a+') as f:
            f.write('\n' + ','.join([str(err) for err in error_list + [np.mean(error_list)]]))
    else:
        with open(error_fpath, 'w+') as f:
            f.write('# errors for {} and mean error'.format(orient_list))
