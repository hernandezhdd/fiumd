from fiumd import _version

# Expose main modules as attributes of fiumd
from . import systems
from . import forces
from . import batch_forces
from . import ensembles
from . import integrator
from . import neighbours
from . import parameters
from . import minimizers
from . import run
from . import utils
from . import wrapper

from . import forcefields

# Optional: if these are legacy or alternative versions,
# don't expose them unless needed by user-facing API
from . import forces_orig
from . import systems_orig
from . import integrator_orig
from . import wrapper_orig

# # Optional: version info
# from ._version import __version__


# __version__ = _version.get_versions()["version"]
