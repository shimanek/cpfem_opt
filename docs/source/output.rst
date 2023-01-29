============
Output Files
============

The optimization script outputs information on each iteration as plaintext in addition to saving stress-strain information as binary numpy arrays. The plotting script save several plots about the optimization run and the final best-fit parameter set.


opt_fea
=======

out_progress.txt
----------------
The main output file.


out_errors.txt
--------------
Separated errors by orientations.


out_time_disp_force_name.npy
------------------------------
Numpy binary for each orientation indicated by ``name``, the nickname in :ref:`orientations`.


temp_expSS.csv
--------------
Truncated experimental stress-strain data if cut down by :ref:`max_strain` parameter.


temp_time_disp_force_name.csv
-------------------------------
Plaintext version of (simulation) time, displacements, and force data from the run indicated by ``name``, the nickname in :ref:`orientations`.


opt_plot
========

out_best_params.txt
-------------------
Plaintext summary of the best parameter set found in :ref:`out_progress.txt`.


res_convergence.png
-------------------
Optimization convergence behavior: cumulative lowest error value as a function of iteration.


res_evaluations.png
-------------------
NxN lower triangular plot (where N is the length of optimizeable parameters in :ref:`params`) showing the sampling of each parameter to be optimized.


res_objective.png
-----------------
NxN lower triangular plot (where N is the length of optimizeable parameters in :ref:`params`) showing the objective function as predicted by the surrogate model over all of parameter space. Note that this involves grid sampling and averaging that may not be appropriate for the true function to be examined.


res_opt_name.png
----------------
All stress-strain curves for the orientation ``name``, the nickname in :ref:`orientations`.


res_single_name.png
-------------------
The best stress-strain curve for the orientation ``name``, the nickname in :ref:`orientations`. Also includes a legend with best-fit parameter values.
