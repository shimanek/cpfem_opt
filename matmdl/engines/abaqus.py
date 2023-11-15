"""
This module contains helper functions for dealing with Abaqus but 
has no Abaqus-specific imports.
"""
from matmdl.parser import uset
import subprocess
import os


def job_run():
    """Run the Abaqus job!"""
    subprocess.run( 
        'abaqus job=' + uset.jobname \
        + ' user=' + uset.umat[uset.umat.find('.')] + '-std.o' \
        + ' cpus=' + str(uset.cpus) \
        + ' double int ask_delete=OFF', shell=True
    )


def job_extract(outname: str):
    """
    Call :py:mod:`matmdl.engines.abaqus_extract` from new shell to extract force-displacement data.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    extractions_script_path = os.path.join(src_dir, "abaqus_extract.py")
    run_string = f'abaqus python {extractions_script_path}'
    subprocess.run(run_string, shell=True)
    os.rename('temp_time_disp_force.csv', 'temp_time_disp_force_{0}.csv'.format(outname))


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

