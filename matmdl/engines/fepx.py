"""
Functions for dealing with FEPX.
"""
from matmdl.core.parser import uset
import numpy as np
import subprocess
import os


def pre_run(next_params, orient, in_opt):
    """Things to do before each run."""
    pass


def run():
    """Starts FEPX, assuming `fepx` and `mpirun` are on system's path."""
    runlog = "temp_run_log"
    if uset.executable_path:
        fepx = uset.executable_path
    else:
        fepx = "fepx"
    subprocess.run(f"mpirun -np ${{SLURM_NTASKS}} {fepx} | tee {runlog}", shell=True)
    # TODO should check uset which should have previously checked for Slurm variables


def prepare():
    """
    Main call to prepare for all runs. Nothing to do for FEPX?
    """
    pass


def extract(outname: str):
    """
    Get time, displacement, force data from simdir.

    Note:
        This does extra work to match the Abaqus format... worth a change?
    """
    # get loading direction from config
    config = _parse_config()
    loading_dir = config["loading_direction"][0]
    strain = float(config["target_strain"][0])

    # extract output data:
    data = np.loadtxt(os.path.join("simulation.sim", "results", "forces", loading_dir+"1"), skiprows=2)
    possible_dirs = ["x", "y", "z"]
    force = data[:, 2+possible_dirs.index(loading_dir)]
    time = data[:, -1]
    time = time / max(time)

    # use uset dimensions if available, fallback to area and assumption of a cube
    if uset.length:
        length_og = uset.length
    else:
        area = data[0,5]
        print("DBG: this should be 1.0:", area)
        length_og = area**(0.5)

    displacement = time * strain * length_og


    time_disp_force = np.stack((time.transpose(), displacement.transpose(), force.transpose()), axis=1)
    header = "time, displacement, force"
    np.savetxt(f"temp_time_disp_force{outname}.csv", time_disp_force, header=header, delimiter=",")


def _parse_config(key=None):
    """read simulation.cfg, return string dictionary of key:[list of values], all lowercase"""
    lines = {}
    with open("simulation.cfg", "r") as f:
        for line in f.readlines():
            if line.strip() and not line.strip().startswith("#"):
                sections = line.strip().split(" ")
                if sections[0] not in lines.keys():
                    lines[sections[0]] = [section.lower() for section in sections[1:]]
                else:
                    lines[sections[0]] = lines[sections[0]] + [section.lower() for section in sections[1:]]

    if key is not None:
        return lines[key]
    else:
        return lines


def has_completed():
    """
    Return ``True`` if Abaqus has finished sucessfully.
    """
    runlog = "temp_run_log"
    if os.path.isfile(runlog):
        try:
            check_line = subprocess.run(f"tail -n2 {runlog} | head -n 1", capture_output=True, check=True, shell=True).stdout.decode("utf-8")
        except subprocess.CalledProcessError:
            raise RuntimeError("FEPX incomplete run")
    else: 
        check_line = ''
    return ('Final step terminated. Simulation completed successfully.' in check_line)


def write_strain(strain: float, jobname: str=None, debug=False):
    """
    Modify boundary conditions in main Abaqus input file to match max strain.
    
    Args:
        strain: signed float used to specify axial displacement
        jobname: ignored, only included to match call signature in abaqus

    Note:
        Relies on finding `target_strain` in simulation.cfg
    """
    fname = "simulation.cfg"
    key = "target_strain"
    max_bound = round(strain, 4) #round to 4 digits

    new_lines = []
    with open(fname, "r+") as f:
        for line in f.readlines():
            if line.strip().startswith(key):
                old_items = line.strip().split(" ")
                # ^ expecting: target_strain <strainValue> <outputFrequency> print_data
                space = " "  # to avoid another set of quotes within the follow f-string
                new_line = line[0:line.find(key)] + f"{old_items[0]} {strain} {space.join(old_items[2:])}\n"
                new_lines.append(new_line)
            else:
                new_lines.append(line)

    with open(f"temp_{fname}", 'w+') as f:
        f.writelines(new_lines)

    if not debug:
        os.remove(fname)
        os.rename('temp_' + fname, fname)
