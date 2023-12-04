"""
module for writing to files
"""
from matmdl.runner import combine_SS
from matmdl.optimizer import update_progress, write_opt_progress


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
    error_fname = 'out_errors.txt'
    if os.path.isfile(error_fname):
        with open(error_fname, 'a+') as f:
            f.write('\n' + ','.join([str(err) for err in error_list + [np.mean(error_list)]]))
    else:
        with open(error_fname, 'w+') as f:
            f.write('# errors for {} and mean error'.format(orient_list))

