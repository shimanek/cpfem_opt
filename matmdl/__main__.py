from matmdl.run import main as run
from matmdl.plot import main as plot

import warnings
warnings.formatwarning = lambda msg, *args, **kwargs: f"{type(args[0]).__name__}: {msg}\n"
# TODO: check that formatting here applies to later use of warnings.warn

run()
plot()
