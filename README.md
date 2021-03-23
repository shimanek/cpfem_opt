# cpfem_opt
A method of optimizing CPFEM parameters using scikit-optimize and plaintext input files.

## Files included:
- opt_fea.py: the main file controlling the optimization process
- opt_plot.py: for plotting results on the cluster
- qopt_: a basic PBS job submission script

## Dependencies:
This requires scikit-optimize, which can be installed to a new Conda environment called `opt` with the following steps:

1. `conda create --name opt`
2. Either `conda activate opt` OR `source activate opt`
3. `conda install -c conda-forge scikit-optimize`
