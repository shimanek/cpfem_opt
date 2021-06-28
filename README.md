# cpfem_opt
A method of optimizing CPFEM parameters using scikit-optimize and plaintext input files.

## Files included:
main:

	- opt_input.py: includes all user input
	- opt_fea.py: 	the main file controlling the optimization process
	- opt_plot.py: 	for plotting results on the cluster
	- qopt_: 		a basic PBS job submission script
	
model_generation:
	- mk_singleCrystalModel.py: limited script to generate Abaqus mesh files for long chains of elements appropriate for single crystal deformation simulations
	- mk_orthoModel.py: 		script to generate Abaqus input files for polycrystalline models for arbitrary sizes of orthorhombic grains within orthorhombic volume elements

## Dependencies:
This requires scikit-optimize, which can be installed to a new Conda environment called `opt` with the following steps. Configuration files are located in `src/install/`.

`conda env create --file=config_simple.yaml`

or by using specific dependency versions, which have been tested:

`conda env create --file=config_strict.yaml`
