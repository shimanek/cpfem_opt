==============
Input Settings
==============

.. role:: py(code)
   :language: python3

The input file ``opt_input.py`` contains many parameters relevant to specifying file names and also optimization controls and options. Some detailed optimizer settings are currently still housed within ``opt_fea.py`` in an attempt to simplify the input file. The inputs are outlined below.


param_list
==========
**list of strings:** Contains hardening parameter names. Values having these names in the file specified as the ``param_file`` variable will be found and modified during optimization.

param_bounds
============
**list of tuples of floats:** Lower and upper bounds on each parameter given in the same index of :ref:`param_list`. Really, these should be combined into one dictionary-type input.

orientations
============
**dictionary:** Structure giving orientation information that allows the optimization scheme to consider offset angles and magnitudes for each loading separately. An annotated structure with offset consideration looks like the following:

.. code-block:: python3

	orientations = {
		'001':{  # nickname string for simulation
			'exp':'exp_Cu-mX-001.csv',  # file name of experimental data
			'offset':{
				'dir_load':(0,0,1),  # loading direction in Miller indices
				'dir_0deg':(0,1,1),  # orthogonal load corresponding to how you choose 0˚ twisting
				'mag_bounds':(0,1),  # magnitude of tilt
				'deg_bounds':(0,90),  # direction of tilt as a right hand twist about dir_load
			}
		},
	}

Note that ``mag_bounds`` and ``deg_bounds`` can be a float instead of a tuple, which will indicate that no optimization will be carried out in terms of this loading's orientation.

Alternatively to specifying an ``orientations`` dictionary, one can pass information on the input file that will be used without modification, such as in the following:

.. code-block:: python3

	orientations = {
		'x':{
			'exp':'exp_W_mX_111_plus15.0deg_towards001.csv',
			'inp':'mat_orient_15_plus2.0deg_towards213.inp'
		},
		'y':{
			'exp':'exp_W_mX_111_plus27.9deg_towards001.csv',
			'inp':'mat_orient_27_plus2.0deg_towards213.inp'
		},
		'z':{
			'exp':'exp_W_mX_111_plus36.5deg_towards001.csv',
			'inp':'mat_orient_36_plus2.0deg_towards213.inp'
		},
	}


param_file
==========
**string:** Name of the file containing the parameters, in Abaqus ``parameter`` format, for the CPFEM run. The values in this file will be modified if listed in :ref:`param_list`.

Since the inputs are in Python, this could be specified semi-automatically by something like :py:`[f for f in os.listdir(os.getcwd()) if f.startswith('mat_param')][0]`, as long as one is sure which single file matches that description; this was useful in an early iteration of the project.


loop_len
========
**int:** Total number of iterations for the optimization scheme. Includes ``n_initial_points`` in the count.


n_initial_points
================
**int:** Number of iterations to be sampled before the optimization's acquisition function takes over determining which areas of parameter space to sample next. Worth experimenting with, but reasonable convergence has been seen with ``n_initial_points`` set to about half or a third of total iterations when ``loop_len``\~100 and :py:`len(param_list)=5`.


large_error
===========
**float:** Backup error value to send to the optimizer for the case of runs which don't finish. The first choice is set in ``opt_fea.py`` as 1.5 * IQR(first few RMSE), where IQR is the interquartile range. This is preferrable since the error returned should be large enough to dissuade the acquisition function from exploring that area of parameter space without being so large as to cause a discontinuity that affects the surrogate model's predictions in other areas of parameter space.


length
======
**float:** Axial length along uniaxial loading direction (y-direction by default) to convert displacements into engineering strains.


area
====
**float:** Model area normal to the uniaxial loading direction to convert forces to engineering strains.


jobname
=======
**string:** Main input file name for the Abaqus job.


recursion_depth
===============
**int:** Maximum number of times that the Abaqus run is restarted with a smaller maximum increment. The factor by which the Abaqus increment is set is given in ``opt_fea.refine_run()`` and is currently 5. If ``recursion_depth=2`` then an initial increment of 1E-2 will be cut to 2E-3 and then to 4E-3 in an attempt to get a converging Abaqus solution before returning to 1E-2 for the next parameter set.


max_strain
==========
**float:** 0 for max exp value, fractional strain (0.01=1%) otherwise


i_powerlaw
==========
**int:** Specifies the interpolation type between experimental data points, which can be based on either:
	``0``: linear interpolation between two relevant points

	``1``: power-law/Holomon fitting of the entire stress-strain curve

Fitting with the Holomon equation is useful for polycrystal data with low resolution (\~ 8 data points per curve).


umat
====
**string:** File name specifying the location of the user material subroutine.


cpus
====
**int:** Number of cores on which to run the Abaqus job.


do_load_previous
================
**boolean:** True if the optimizer should load the previous ``out_progress`` file. Currently, reloading requires that all output was strictly within the current bounds specified in :ref:`param_bounds`, although this could be autmatically determined in the future if needed. Note that, for clarity, previous runs are given negative iteration numbers in the new ``out_progress`` file.


grain_size_name
===============
**string:** Deprecated legend key.


title
=====
**string:** Optional plot title.


param_additional_legend
=======================
**list of strings:** Extra parameters in addition to those in :ref:`param_list` that will be plotted in the single stress-strain plots showing the best-fit parameter set and its comparable experimental curve. Useful if one hardening parameter has been manually set in :ref:`param_file` but is still of interest to the plotted results.
