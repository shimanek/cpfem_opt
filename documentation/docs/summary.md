# Overview


`matmdl` is a package implementing a Bayesian optimization [framework](https://scikit-optimize.github.io/stable/) using Gaussian process regression to parameterize the hardening models within crystal plasticity finite element method calculations, which are run through a [user material subroutine](https://www.researchgate.net/profile/Frank-Richter-5/post/Can_anyone_help_me_te_implement_properly_Huangs_UMAT_for_single_crystal_plasticity_on_Abaqus/attachment/5ca8491b3843b01b9b97ef84/AS%3A744577992499201%401554532635182/download/Huang+-+MECH+178.pdf) integrated with Abaqus. 
Its main features are that it:

* Handles multiple input data sets simultaneously, e.g. mutliple orientations in single crystal deformation or multiple grain sizes
* Uses gradient-free optimization for the multi-dimensional parameter space describing slip system level material properties
* Wraps around an existing crystal plasticity implementation using Abaqus as the finite element solver[^1]
* Considers slight offset orientations in the case of single crystal plasticity, which may go unquantified and unreported in most experimental data sets in the literature


# Basic Usage

Each run requires an `input.toml` file to specify the most common input options. 
The runnable modes are each modules:

* `matmdl.run` starts an optimization run
* `matmdl.plot` plots the results of a complete or incomplete optimization run

The input options are detailed in [input](input.md).


[^1]: The optional use of [FEPX](https://fepx.info) as an alternate CPFEM engine is in the process of being added.
