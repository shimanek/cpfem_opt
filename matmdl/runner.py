from matmdl.engines.abaqus import job_run, check_complete, job_extract
from matmdl.parser import uset
from matmdl.parallel import Checkout
from typing import Union
import numpy as np
import subprocess
import os


def get_first(opt: object, in_opt: object) -> None:
    """
    Run one simulation so its output dimensions can later inform the shape of output data.
    """
    job_run()
    if not check_complete():
        refine_run()
    job_extract('initial')


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
    for f in job_files:
        if os.path.isdir(f):
            os.rmdir(f)
        else:
            os.remove(f)


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
