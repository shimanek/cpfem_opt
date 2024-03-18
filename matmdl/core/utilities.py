"""
Utility functions that fit nowhere else.
These should have no `matmdl` dependencies.
"""
import numpy as np
from numpy.linalg import norm
import warnings
from typing import Union
import time


def log(message: str):
    """append message to local log file with time stamp"""
    with open("out_log.txt", "a+") as f:
        f.write(f"{time.time():.2f}: {message}\n")


def msg(message: str):
    """broadcast message to stdout if not run with -0"""
    if __debug__:
        print(message, flush=True)


def warn(message: str, warn_type=UserWarning):
    """Raise warning with consistent formatting"""
    warnings.formatwarning = lambda msg, warn_type, *args, **kwargs: f"{warn_type.__name__}: {msg}\n"
    warnings.warn(message, warn_type)


def unit_vector(vector: 'vector') -> 'vector':
    """Gives a normalized vector using ``numpy.linalg.norm``."""
    return vector/norm(vector)


def as_float_tuples(list_of_tuples: list[tuple[Union[int,float]]]) -> list[tuple[float]]:
    """
    Make sure tuples contain only floats.

    Take list of tuples that may include ints and return list of tuples containing only floats. Useful for optimizer param bounds since type of input determines type of param guesses. Skips non-tuple items in list.

    Args:
        list_of_tuples: Tuples in this list may contain a mix of
            floats and ints.
    Returns:
        The same list of tuples containing only floats.

    """
    new_list = []
    prec = 10  # decimal places in scientific notation
    def sigfig(val):
        return float(('%.' + str(prec) + 'e') % val)
    for old_item in list_of_tuples:
        if isinstance(old_item, tuple):
            new_item = tuple(map(sigfig, old_item))
        else:
            new_item = old_item
        new_list.append(new_item)
    return new_list


def round_sig(x: float, sig: int=4) -> float:
    if x == 0.0: 
        return 0.
    return round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)
