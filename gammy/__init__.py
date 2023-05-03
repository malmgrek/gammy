"""Importable modules in the package

.. autosummary::
   :toctree: _autosummary

   arraymapper
   formulae
   models
   plot
   utils

"""

import importlib.metadata

from .arraymapper import ArrayMapper, x
from .formulae import *
from .models import *
from .utils import *

try:
    from .plot import *
except ImportError:
    # Allow import plot fail in environments where matplotlib is not installed.
    pass

__version__ = importlib.metadata.version("gammy")
