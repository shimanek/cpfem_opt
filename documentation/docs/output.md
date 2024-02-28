# Output Files

The optimization script outputs information on each iteration as plaintext in addition to saving stress-strain information as binary numpy arrays. 
The plotting script save several plots about the optimization run and the final best-fit parameter set.


## run

* `out_progress.txt`: csv with 1-line header and columns of unix time in ns, then parameter set values
* `out_errors.txt`: csv with 1-line header and columns of the error value for each sample and the overall error as the last column
* `out_time_disp_force_name.npy`: Numpy binary for each orientation indicated by ``name``, the nickname in :ref:`orientations`.
* `temp_expSS.csv`: Truncated experimental stress-strain data if cut down by `max_strain` parameter.
* `temp_time_disp_force_name.csv`: plaintext version of (simulation) time, displacements, and force data from the run indicated by `name`, the nickname in the `orientations` section of the input file.


## plot

* `out_best_params.txt`: Plaintext summary of the best parameter set found in :ref:`out_progress.txt`.
* `res_convergence.png`: Optimization convergence behavior: cumulative lowest error value as a function of iteration.
* `res_evaluations.png`: NxN lower triangular plot (where N is the length of optimizeable parameters in :ref:`params`) showing the sampling of each parameter to be optimized.
* `res_objective.png`: NxN lower triangular plot (where N is the length of optimizeable parameters in :ref:`params`) showing the objective function as predicted by the surrogate model over all of parameter space. Note that this involves grid sampling and averaging that may not be appropriate for the true function to be examined.
* `res_opt_name.png`: All stress-strain curves for the orientation ``name``, the nickname in :ref:`orientations`.
* `res_single_name.png`: The best stress-strain curve for the orientation ``name``, the nickname in :ref:`orientations`. Also includes a legend with best-fit parameter values.
