# API: Models

These files are runnable as modules, e.g. `python -m matmdl.models.mk_orthoModel`. The long name hasn't been an issue due to how infrequently they are used in comparison to the other possible calls within this program.

The [monocrystal model](#matmdl.models.mk_singleModel) writes the Abaqus input files for a single element with tilting face boundary conditions that allow for lattice reorientation expected to occur during tensile deformation. More details can be found in the related publication.[^1]

The [polycrystal model](#matmdl.models.mk_orthoModel) writes Abaqus input files specifying multiple orthorhombic grains within an orthorhombic volume element. Calling this module will walk one through the options interactively.

::: matmdl.models

[^1]: [doi: 10.1016/j.commatsci.2024.112879](https://doi.org/10.1016/j.commatsci.2024.112879)
