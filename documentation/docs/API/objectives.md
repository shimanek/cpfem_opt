# API: Objectives

These files determine the error metrics based on comparison between simulated and experimental stress-strain curves. [Calculate](#matmdl.objectives.calculate) is the main call for individual runs. It compares interpolated curves and gives a scalar error metric based on combinations of value and slope differences. [Combine](#matmdl.objectives.combine_variance) and similar files (`combine_*.py`) combine those scalar errors across different datasets for a single global objective function when multiple datasets are present.

::: matmdl.objectives
