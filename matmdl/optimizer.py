class InOpt:
    """
    Stores information about the optimization input parameters.

    Since the hardening parameters and orientation parameters are fundamentally
    different, this object stores information about both in such a way that they
    can still be access independently.

    Args:
        orientations (dict): Orientation information directly from ``opt_input``.
        param_list (list): List of material parameters to be optimized.
        param_bounds (list): List of tuples describing optimization bounds for
            variables given in ``param_list``.

    Attributes:
        orients (list): Nickname strings defining orientations, as given
            in :ref:`orientations`.
        material_params (list): Parameter names to be optimized, as in :ref:`orientations`.
        material_bounds (list): Tuple of floats defining bounds of parameter in the same
            index of ``self.params``, again given in :ref:`orientations`.
        orient_params (list): Holds orientation parameters to be optimized, or single
            orientation parameters if not given as a tuple in :ref:`orientations`.
            These are labeled ``orientationNickName_deg`` for the degree value of the
            right hand rotation about the loading axis and ``orientationNickName_mag``
            for the magnitude of the offset.
        orient_bounds (list): List of tuples corresponding to the bounds for the parameters
            stored in ``self.orient_params``.
        params (list): Combined list consisting of both ``self.material_params`` and
            ``self.orient_params``.
        bounds (list): Combined list consisting of both ``self.material_bounds`` and
            ``self.orient_bounds``.
        has_orient_opt (dict): Dictionary with orientation nickname as key and boolean
            as value indicating whether slight loading offsets should be considered
            for that orientation.
        fixed_vars (dict): Dictionary with orientation nickname as key and any fixed
            orientation information (``_deg`` or ``_mag``) for that loading orientation
            that is not going to be optimized.
        offsets (list): List of dictionaries containing all information about the offset
            as given in the input file. Not used/called anymore?
        num_params_material (int): Number of material parameters to be optimized.
        num_params_orient (int): Number of orientation parameters to be optimized.
        num_params_total (int): Number of parameters to be optimized in total.

    Note:
        Checks if ``orientations[orient]['offset']['deg_bounds']``
        in :ref:`orientations` is a tuple to determine whether
        orientation should also be optimized.
    """
    # TODO: check if ``offsets`` attribute is still needed.
    def __init__(self, orientations, params):
        """Sorted orientations here defines order for use in single list passed to optimizer."""
        self.orients = sorted(orientations.keys())
        self.params, self.bounds, \
        self.material_params, self.material_bounds, \
        self.orient_params, self.orient_bounds \
            = ([] for i in range(6))
        for param, bound in params.items():
            if type(bound) in (list, tuple):
                self.material_params.append(param)
                self.material_bounds.append(bound)
            elif type(bound) in (float, int):
                write_params(uset.param_file, param, float(bound))
            else:
                raise TypeError('Incorrect bound type in input file.')
        
        # add orientation offset info:
        self.offsets = []
        self.has_orient_opt = {}
        self.fixed_vars = {}
        for orient in self.orients:
            if 'offset' in orientations[orient].keys():
                self.has_orient_opt[orient] = True
                self.offsets.append({orient:orientations[orient]['offset']})
                # ^ saves all info (TODO: check if still needed)

                # deg rotation *about* loading orientation:
                if isinstance(orientations[orient]['offset']['deg_bounds'], tuple):
                    self.orient_params.append(orient+'_deg')
                    self.orient_bounds.append(orientations[orient]['offset']['deg_bounds'])
                else:
                    self.fixed_vars[(orient+'_deg')] = orientations[orient]['offset']['deg_bounds']

                # mag rotation *away from* loading:
                if isinstance(orientations[orient]['offset']['mag_bounds'], tuple):
                    self.orient_params.append(orient+'_mag')
                    self.orient_bounds.append(orientations[orient]['offset']['mag_bounds'])
                else:
                    self.fixed_vars[(orient+'_mag')] = orientations[orient]['offset']['mag_bounds']

            else:
                self.has_orient_opt[orient] = False
        
        # combine material and orient info into one ordered list:
        self.params = self.material_params + self.orient_params
        self.bounds = as_float_tuples(self.material_bounds + self.orient_bounds)
        
        # descriptive stats on input object:
        self.num_params_material = len(self.material_params)
        self.num_params_orient = len(self.orient_params)
        self.num_params_total = len(self.params)
