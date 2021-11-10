"""Importable modules in the package

.. autosummary::
   :toctree: _autosummary

   arraymapper
   formulae
   models
   plot
   utils

"""

from .__version__ import (
    __author__,
    __copyright__,
    __description__,
    __license__,
    __title__,
    __url__,
    __version__
)

from .arraymapper import ArrayMapper, x
from .formulae import *
from .models import *
from .utils import *

try:
    from .plot import *
except ImportError:
    # Allow import plot fail in environments where matplotlib is not installed.
    pass
