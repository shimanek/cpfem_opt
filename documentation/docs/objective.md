# Objective

The optimization procedure relies on a definition of goodness of fit between the experimental and simulated stress-strain curves. 
Here, the objective function is defined in [`objectives/rmse.py`][matmdl.objectives.rmse] , which currently uses values based on both stress differences and slope differences. 
These are combined based on the input value `slope_weight`, with 0.0 being all stress-based and 1.0 being entirely slope-based.

The values themselves are root mean square differences/errors (RMSEs) over the entire deformation, although limits can be applied to ignore certain regions. The RMSEs are normalized by the mean of the observations, meaning the average of the interpolated stress or strain values; therefore, these should not depend on the number of experimental data points. Note that something like an elastomer, where the slope of the stress-strain curve is around zero for large strains, would ruin the purpose of this normalization procedure.

For multiple samples or single crystal orientations, the errors are combined by taking their mean. So each sample's error value, itself a combination of stress and slope errors, has equal weight during the final combination to a single-valued minimization objective.

Unlike changing parameter bounds, a change in the objective function changes the objective manifold itself, meaning the optimization strategy may have an easier or harder time finding minima and that those minima will likely be different than before the change.
