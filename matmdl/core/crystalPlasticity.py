"""
This module contains functions relevant to the application of Huang's
crystal plasticity subroutine.
"""
import numpy as np
from numpy.linalg import norm
from scipy.optimize import root

from matmdl.core.utilities import unit_vector
from matmdl.core.parser import uset
from matmdl.core import writer as writer


def do_orientation_inputs(next_params, orient, in_opt):
    """
    Get and write new orientation information, if necessary.

    If input file is in orientation structure, uses that.
    """
    if not in_opt.has_orient_opt[orient] and not 'inp' in uset.orientations[orient]:
        return

    if in_opt.has_orient_opt[orient]:
        orient_components = get_orient_info(next_params, orient, in_opt)
        writer.write_input_params('mat_orient.inp', orient_components['names'], orient_components['values'])
    else:
        if 'inp' in uset.orientations[orient]:
            shutil.copy(uset.orientations[orient]['inp'], 'mat_orient.inp')


def get_orient_info(
        next_params: list, 
        orient: str,
        in_opt: object,
    ) -> dict:
    """
    Get components of orientation-defining vectors and their names
    for substitution into the orientation input files.

    Args:
        next_params: Next set of parameters to be evaluated
            by the optimization scheme.
        orient: Index string for dictionary of input
            orientations specified in :ref:`orientations`.
    """
    dir_load = np.asarray(uset.orientations[orient]['offset']['dir_load'])
    dir_0deg = np.asarray(uset.orientations[orient]['offset']['dir_0deg'])

    if (orient+'_mag' in in_opt.params):
        index_mag = in_opt.params.index(orient+'_mag')
        angle_mag = next_params[index_mag]
    else:
        angle_mag = in_opt.fixed_vars[orient+'_mag']

    if (orient+'_deg' in in_opt.params):
        index_deg = in_opt.params.index(orient+'_deg')
        angle_deg = next_params[index_deg]
    else:
        angle_deg = in_opt.fixed_vars[orient+'_deg']

    col_load = unit_vector(np.asarray(dir_load))
    col_0deg = unit_vector(np.asarray(dir_0deg))
    col_cross = unit_vector(np.cross(col_load, col_0deg))

    basis_og = np.stack((col_load, col_0deg, col_cross), axis=1)
    rotation = _mk_x_rot(np.deg2rad(angle_deg))
    basis_new = np.matmul(basis_og, rotation)
    dir_to = basis_new[:,1]

    if __debug__:  # write angle_deg rotation info
        dir_load = dir_load / norm(dir_load)
        dir_to = dir_to / norm(dir_to)
        dir_0deg = dir_0deg / norm(dir_0deg)
        with open('out_debug.txt', 'a+') as f:
            f.write('orientation: {}'.format(orient))
            f.write('\nbasis OG: \n{0}'.format(basis_og))
            f.write('\n')
            f.write('\nrotation: \n{0}'.format(rotation))
            f.write('\n')
            f.write('\nbasis new: \n{0}'.format(basis_new))
            f.write('\n\n')
            f.write('dir_load: {0}\tdir_to: {1}\n'.format(dir_load, dir_to))
            f.write('angle_deg_inp: {0}\n'.format(angle_deg))
            f.write('all params: {}'.format(next_params))

    sol = get_offset_angle(dir_load, dir_to, angle_mag)
    dir_tot = dir_load + sol * dir_to
    dir_ortho = np.array([1, 0, -dir_tot[0]/dir_tot[2]])

    if __debug__: # write final loading orientation info
        angle_output = np.arccos(np.dot(dir_tot, dir_load)/(norm(dir_tot)*norm(dir_load)))*180./np.pi
        with open('out_debug.txt', 'a+') as f:
            f.write('\ndir_tot: {0}'.format(dir_tot))
            f.write('\ndir_ortho: {0}'.format(dir_ortho))
            f.write('\nangle_mag_input: {}\tangle_mag_output: {}'.format(angle_mag, angle_output))
            f.write('\n\n')

    component_names = ['x1', 'y1', 'z1', 'u1', 'v1', 'w1']
    component_values = list(dir_ortho) + list(dir_tot)

    return {'names':component_names, 'values':component_values}


def _mk_x_rot(theta: float) -> 'vector':
    """
    Generates rotation matrix for theta (radians) clockwise rotation 
    about first column of 3D basis when applied from right.
    """
    rot = np.array([[1,             0,              0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta),  np.cos(theta)]])
    return rot


def get_offset_angle(
    direction_og: 'vector', 
    direction_to: 'vector', 
    angle: float) -> float:
    """
    Iterative solution for finding vectors tilted toward other vectors.

    Args:
        direction_og: Real space vector defining
            the original direction to be tilted away from.
        direction_to: Real space vector defining
            the direction to be tilted towards.
        angle: The angle, in degrees, by which to tilt.

    Returns:
        float:
            a scalar multiplier such that the angle between ``direction_og``
            and ``sol.x`` * ``direction_to`` is ``angle``.

    """
    def _opt_angle(
        offset_amt: float, 
        direction_og: 'vector',
        direction_to: 'vector',
        angle: float):
        """
        Angle difference between original vector and new vector, which is
        made by small offset toward new direction.  Returns zero when offset_amt 
        produces new vector at desired angle.  Uses higher namespace variables so 
        that the single argument can be tweaked by optimizer.
        """
        direction_new = direction_og + offset_amt * direction_to
        angle_difference = \
            np.dot(direction_og, direction_new) / \
            ( norm(direction_og) * norm(direction_new) ) \
            - np.cos(np.deg2rad(angle))
        return angle_difference

    sol = root(_opt_angle, 0.01, args=(direction_og, direction_to, angle), tol=1e-10).x[0]
    return sol


def param_check(param_list: list[str]):
    """
    True if tau0 >= tauS

    In theory, tau0 should always come before tauS, even though it doesn't make a difference
    mathematically/practically. Function checks for multiple systems if numbered in the form
    ``TauS``, ``TauS1``, ``TauS2`` and ``Tau0``, ``Tau01``, ``Tau02``.
    """
    # TODO: ck if it's possible to satisfy this based on mat_params and bounds, raise helpful error
    tau0_list, tauS_list = [], []
    for sysnum in ['', '1', '2']:
        if ('TauS'+sysnum in param_list) or ('Tau0'+sysnum in param_list):
            f1 = open(uset.param_file, 'r')
            lines = f1.readlines()
            for line in lines:
                if line.startswith('Tau0'+sysnum): 
                    tau0_list.append(float(line[7:]))
                if line.startswith('TauS'+sysnum): 
                    tauS_list.append(float(line[7:]))
            f1.close()
    is_bad = any([(tau0 >= tauS) for tau0, tauS in zip(tau0_list, tauS_list)])
    return is_bad
