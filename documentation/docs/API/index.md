# General Layout

The main runnable modules, `run` and `plot`, rely on the following module groups:

- [Core](core.md): contains modules used for most applications
- [Engines](engines.md): interfaces with the FEA program
- [Objectives](objectives.md): provides objective functions for the optimization
- [Models](models.md): constructs geometric models for use in the FEA


Note that the [recalculate module](objectives.md#matmdl.objectives.recalculate) is also runnable. 

&nbsp;

# Detailed Layout

The source files are organized into several modules, only a few of which can be accessed from outside. Parts of the source expected to be added to as separate, selectable options are placed into folders with plural nomenclature. Within those, the `__init__.py` files are used to specify which versions are used by the rest of the code. The `core` group provides general functionality that is not expected to be added to, only modified or replaced.

Currently, there is overspecificity within the core/crystalPlasticity module, some of which should be separated to Abaqus-specific files, providing a more general interface for functions of interest to slip system and orientation related functions.

The source layout is the following:
```sh
├── __init__.py
├── __main__.py
├── core
│   ├── __init__.py
│   ├── __pycache__
│   ├── crystalPlasticity.py
│   ├── experimental.py
│   ├── optimizer.py
│   ├── parallel.py
│   ├── parser.py
│   ├── runner.py
│   ├── state.py
│   ├── utilities.py
│   └── writer.py
├── engines
│   ├── __init__.py
│   ├── __pycache__
│   ├── abaqus.py
│   ├── abaqus_extract.py
│   └── fepx.py
├── models #(1)
│   ├── __init__.py
│   ├── __pycache__
│   ├── mk_orthoModel.py
│   └── mk_singleModel.py
├── objectives
│   ├── __init__.py
│   ├── __main__.py
│   ├── __pycache__
│   ├── calculate.py
│   ├── combine_mean.py
│   ├── combine_variance.py
│   └── recalculate.py #(2)
├── plot.py
└── run.py
```

1. These modules are runnable, e.g. `python -m matmdl.models.mk_orthoModel`
2. This module is runnable: `python -m matmdl.objectives.recalculate`
