"""
This module contains helper functions for dealing with Abaqus but 
has no Abaqus-specific imports.
"""
from matmdl.core.parser import uset
import subprocess
import os


def run():
    """Run the Abaqus job!"""
    subprocess.run( 
        'abaqus job=' + uset.jobname \
        + ' user=' + uset.umat[:uset.umat.find('.')] + '-std.o' \
        + ' cpus=' + str(uset.cpus) \
        + ' double int ask_delete=OFF', shell=True
    )


def prepare():
    """
    Main call to prepare for all runs.
    """
    load_subroutine()


def load_subroutine():
    """
    Compile the user subroutine uset.umat as a shared library in the directory.
    """
    try:
        os.remove('libstandardU.so')
    except FileNotFoundError:
        pass
    try:
        os.remove(f'{uset.umat[:uset.umat.find(".")]}-std.o')
    except FileNotFoundError:
        pass

    subprocess.run('abaqus make library=' + uset.umat, shell=True)


def extract(outname: str):
    """
    Call :py:mod:`matmdl.engines.abaqus_extract` from new shell to extract force-displacement data.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    extractions_script_path = os.path.join(src_dir, "abaqus_extract.py")
    run_string = f'abaqus python {extractions_script_path}'
    subprocess.run(run_string, shell=True)
    os.rename('temp_time_disp_force.csv', 'temp_time_disp_force_{0}.csv'.format(outname))


def has_completed():
    """
    Return ``True`` if Abaqus has finished sucessfully.
    """
    stafile = uset.jobname + '.sta'
    if os.path.isfile(stafile):
            last_line = str(subprocess.check_output(['tail', '-1', stafile]))
    else: 
        last_line = ''
    return ('SUCCESSFULLY' in last_line)


def write_strain(jobname: str, strain: float):
    """
    Modify boundary conditions in main Abaqus input file to match max strain.
    
    Args:
        jobname: Filename for main Abaqus job -- unique to 
            orientation if applicable.
        strain: signed float used to specify axial displacement

    Note:
        Relies on finding ``RP-TOP`` under ``*Boundary`` keyword in main
        input file.
    """
    # input file:
    max_bound = round(strain * uset.length, 4) #round to 4 digits

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