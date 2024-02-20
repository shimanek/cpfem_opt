# Input Details

## Experimental Data

`matmdl` expects experimental data as a file of comma separated values, with the first column being engineering strain as a fraction (not percent) and the second column being engineering stress. Stress units should be consistent with those expected of the simulation. In the examples here, those are MPa.


## Input Options

The `input.toml` file contains all of the most common user options for runs. However, several details are not described by this file. For example, the surrogate model details are specified in `optimizer.py` and not at all in the input file. This may change in future iterations if needed, but since the repo is installed in editable mode, changes in the source code are straightforward.

Also note that some common options have default values, defined in [`parser.py`](API/core.md#matmdl.parser) in the `input_reqs` dictionary.

Below are details of setting, grouped by toml heading.


## params
Contains hardening parameter names as keys. 
Values having these names in the file specified as the ``param_file`` variable will be found and modified during optimization. 
Values are lower and upper bounds on each parameter, in a tuple.
If the dictionary value is a single number, that parameter value is written to the ``param_file`` and not cosidered during optimization, as in the first value in the following example:


```toml
[params]
Tau0 = 1.5 # (1)
H0 = [200, 500]
TauS = [10, 100]
```

1. This value is written to the input files but is not part of the optimization.


## orientations
Structure giving orientation information that allows the optimization scheme to consider offset angles and magnitudes for each loading separately. For example, optimization of [001] orientation single crystal, in comparison to experimental data in the file `exp_Cu-mX-001.csv`, the input settings might look like this:

```toml
[[orientations]]
name = '001' # (1)
exp = 'exp_Cu-mX-001.csv'
inp = 'mat_orient_100.inp'
[orientations.offset]
dir_load = [0,0,1]
dir_0deg = [0,1,1]
mag_bounds = [0,1]
deg_bounds = 0
```

1. This is a string used for internal identification of the sample.

Note that when `[orientations.offset]` is populated, the `inp` file option is not needed since it will be overwritten. Alternatively, if you have a fixed input file, then none of the `[orientations.offset]` is needed. 

Offset information:

* **dir_load**: Loading direction in 3-tuple Miller indices.

* **dir_0deg**: Non-collinear direction used to define a zero of the rotation *about* the loading direction.

* **mag_bounds**: Degree range for allowable tilt of `dir_load`.

* **deg_bounds**: Degree range for allowable twist about `dir_load`.


## run

* **param_file**: Name of the file containing the parameters, in Abaqus ``parameter`` format, for the CPFEM run. The values in this file will be modified if listed in :ref:`params`. 

* **loop_len**: Total number of iterations for the optimization scheme. Includes ``n_initial_points`` in the count.

* **n_initial_points**: Number of iterations to be sampled before the optimization's acquisition function takes over determining which areas of parameter space to sample next. Worth experimenting with, but reasonable convergence has been seen with ``n_initial_points`` set to about half or a third of total iterations when ``loop_len``\~100 and :py:`len(param_list)=5`.

* **large_error**: Backup error value to send to the optimizer for the case of runs which don't finish. The first choice is set in ``opt_fea.py`` as 1.5 * IQR(first few RMSE), where IQR is the interquartile range. This is preferrable since the error returned should be large enough to dissuade the acquisition function from exploring that area of parameter space without being so large as to cause a discontinuity that affects the surrogate model's predictions in other areas of parameter space.

* **length**: Axial length along uniaxial loading direction (y-direction by default) to convert displacements into engineering strains. ToDo: find automatically from mesh.

* **area**: Model area normal to the uniaxial loading direction to convert forces to engineering strains. ToDo: find automatically from mesh.

* **jobname**: Main input file name for the Abaqus job.

* **recursion_depth**: Maximum number of times that the Abaqus run is restarted with a smaller maximum increment. The factor by which the Abaqus increment is set is given in ``opt_fea.refine_run()`` and is currently 5. If ``recursion_depth=2`` then an initial increment of 1E-2 will be cut to 2E-3 and then to 4E-3 in an attempt to get a converging Abaqus solution before returning to 1E-2 for the next parameter set.

* **max_strain**: Maximum strain to consider from the experimental data and therefore a maximum strain to run the CPFEM calculations until. Set 0 for max experimental value, or use fractional strain (0.01=1%) otherwise.

* **i_powerlaw**: Specifies the interpolation type. Defualt is linear (`0`), good for fine resolution experimental data. Also available is power-law (`1`), which is useful for polycrystal data with low resolution (\~ 8 data points per curve).

* **umat**: File name specifying the location of the user material subroutine.

* **cpus**: Number of cores on which to run the Abaqus job.

* **do_load_previous**: True if the optimizer should load the previous ``out_progress`` file. Currently, reloading requires that all output was strictly within the current bounds specified in :ref:`params` as ``params.values()``. Note that, for clarity, previous runs are given negative iteration numbers in the new ``out_progress`` file. ToDo: automatically filter through output and only reload entries that fall within current parameter bounds.


## plot

* **grain_size_name**: Deprecated legend key.

* **title**: Optional plot title.

* **param_additional_legend**: Extra parameters in addition to those in :ref:`params` that will be plotted in the single stress-strain plots showing the best-fit parameter set and its comparable experimental curve. Useful if one hardening parameter has been manually set in :ref:`param_file` but is still of interest to the plotted results.
