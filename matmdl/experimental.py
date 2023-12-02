"""
Contains the class for extracting and storing experimental data
from plain text inputs for comparison to iterative solution attempts.
"""
from matmdl.parser import uset
import numpy as np


class ExpData():
    """
    Loads and stores experimental data.

    Attributes:
        data (dict): Indexed by orientation name defined in :ref:`orientations`, 
            with values of max strain (internal: ``_max_strain``) and ``raw``, 
            which houses the experimental stress strain data truncated by max strain.

    Note:
        Experimental stress-strain data are expected as plaintext in two columns:
        strain (unitless), and stress (matching the CPFEM inputs, often MPa).

    """
    def __init__(self, orientations: dict):
        self.data = {}
        for orient in orientations.keys():
            expname = orientations[orient]['exp']
            # orientname = orientations[orient]['inp']
            jobname = uset.jobname + '_{0}.inp'.format(orient)
            min_strain, max_strain = self._get_bounds(expname, orient)
            raw = self._get_SS(expname, min_strain, max_strain)
            sgn = -1 if uset.is_compression else 1
            self._write_strain_inp(jobname, sgn*max_strain)
            self.data[orient] = {
                'max_strain': max_strain,
                'min_strain': min_strain,
                'raw': raw,
            }

    def _load(self, fname: str):
        """
        Load original experimental stress-strain data and order it by strain.

        Args:
            fname: Filename for experimental stress-strain data
        """
        original_SS = np.loadtxt(fname, skiprows=1, delimiter=',')
        order = -1 if uset.is_compression else 1
        original_SS = original_SS[original_SS[:,0].argsort()][::order]
        return original_SS

    def _get_bounds(self, fname: str, orient: str):
        """get limiting bounds"""
        mins = []
        maxes = []

        # orientation limits:
        if "min_strain" in uset.orientations[orient].keys():
            mins.append(float(uset.orientations[orient]["min_strain"]))
        if "max_strain" in uset.orientations[orient].keys():
            maxes.append(float(uset.orientations[orient]["max_strain"]))

        # global limits
        if hasattr(uset, "min_strain"):
            mins.append(uset.min_strain)
        if hasattr(uset, "max_strain"):
            if uset.max_strain != 0.0:
                maxes.append(uset.max_strain)

        # data limits
        data = np.sort(np.loadtxt(fname, skiprows=1, delimiter=',' )[:,0])
        mins.append(data[0])
        maxes.append(data[-1])

        # get limiting bounds to use
        if uset.is_compression:  # negative numbers
            min_use = min(mins)
            max_use = max(maxes)
        else:
            min_use = max(mins)
            max_use = min(maxes)

        return min_use, max_use


    def _get_max_strain(self, fname: str, orient: str):
        """
        Take either user max strain or file max strain.
        
        Args:
            fname: Filename for experimental stress-strain data
        """
        if float(uset.max_strain) == 0.0:
            if uset.is_compression is True:
                max_strain = min(np.loadtxt(fname, skiprows=1, delimiter=',' )[:,0])
            else:
                max_strain = max(np.loadtxt(fname, skiprows=1, delimiter=',' )[:,0])
        else:
            max_strain = uset.max_strain if not uset.is_compression else (-1 * uset.max_strain)
        return max_strain

    def _get_min_strain(self, fname: str):
        """
        Take either user min strain or minimum of experimental strain in file `fname`

        Args:
            fname: Filename for experimental stress-strain data
        """
        if float(uset.min_strain) == 0.0:
            if uset.is_compression is True:
                min_strain = max(np.loadtxt(fname, skiprows=1, delimiter=',' )[:,0])
            else:
                min_strain = min(np.loadtxt(fname, skiprows=1, delimiter=',' )[:,0])
        else:
            min_strain = uset.min_strain if not uset.is_compression else (-1 * uset.min_strain)
        return min_strain

    def _get_SS(self, fname: str, _min_strain: float, _max_strain: float):
        """
        Limit experimental data to within min_strain to max_strain. 
        
        Args:
            fname: Filename for experimental stress-strain data
        """
        expSS = self._load(fname)

        if not _max_strain == 0.0:
            expSS = expSS[expSS[:,0] <= _max_strain, :]
        if not _min_strain == 0.0:
            expSS = expSS[expSS[:,0] >= _min_strain, :]

        np.savetxt('temp_expSS.csv', expSS, delimiter=',')
        return expSS

    def _write_strain_inp(self, jobname: str, strain: float):
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
