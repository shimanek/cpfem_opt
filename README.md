# cpfem_opt
A method of optimizing CPFEM parameters using scikit-optimize and plaintext input files.

## Files included:
- opt_input.py: includes all user input
- opt_fea.py: 	the main file controlling the optimization process
- opt_plot.py: 	for plotting results on the cluster
- qopt_: 		a basic PBS job submission script

## Dependencies:
This requires scikit-optimize, which can be installed to a new Conda environment called `opt` with the following steps:

`conda env create --file=config_simple.yaml`

or by using specific dependency versions, which have been tested:

`conda env create --file=config_strict.yaml`
