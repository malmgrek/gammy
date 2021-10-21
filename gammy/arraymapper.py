"""Arraymapper module

.. rubric:: Objects

.. autosummary::
   :toctree:

   x

"""

from functools import wraps
from typing import Callable

from gammy.utils import compose


class ArrayMapper:
    """Callable input mapping object that obey arithmetic operations

    When working with basis function regression, when building the
    design matrix, one needs the information 'how eact term maps input array'.
    This class helps writing human-readable model definitions that explicitly
    show how inputs are mapped. For example,

    .. code-block:: python

       formula = a(x[:, 0]) * x[:, 1] + x[:, 2] ** 2

    is a valid definition as long as ``a`` is an instance of
    :class:`gammy.Formula` and ``x`` is an instance of :class:`ArrayMapper`.

    TODO: Are some natural methods are still missing?

    """

    def __init__(self, function=lambda t: t):

        self.function = function
        """Wrapped function"""

        return

    def __call__(self, *args, **kwargs):
        """Call the wrapped function

        """
        return self.function.__call__(*args, **kwargs)

    def __getitem__(self, key: int) -> "ArrayMapper":
        """Access index as in a NumPy Array

        """
        return ArrayMapper(
            lambda t: t.__getitem__(key)
        )

    def __add__(self, other) -> "ArrayMapper":
        """Addition of arraymappers

        Parameters
        ----------
        other : ArrayMapper

        """
        return ArrayMapper(
            lambda t: self.function(t).__add__(other.function(t))
        )

    def __sub__(self, other) -> "ArrayMapper":
        """Subtraction of arraymappers

        Parameters
        ----------
        other : ArrayMapper

        """
        return ArrayMapper(
            lambda t: self.function(t).__sub__(other.function(t))
        )

    def __mul__(self, other) -> "ArrayMapper":
        """Multiplication of arraymappers

        Parameters
        ----------
        other : ArrayMapper

        """
        return ArrayMapper(
            lambda t: self.function(t).__mul__(other.function(t))
        )

    def __truediv__(self, other) -> "ArrayMapper":
        """Division of arraymappers

        Parameters
        ----------
        other : ArrayMapper

        """
        return ArrayMapper(
            lambda t: self.function(t).__truediv__(other.function(t))
        )

    def __pow__(self, n: float) -> "ArrayMapper":
        """Raise an arraymapper to a power

        """
        return ArrayMapper(
            lambda t: self.function(t).__pow__(n)
        )

    def __neg__(self) -> "ArrayMapper":
        """Negation operation

        """
        return ArrayMapper(
            lambda t: self.function(t).__neg__()
        )

    def ravel(self) -> "ArrayMapper":
        """Imitates the behavior of NumPy ravel method

        """
        return ArrayMapper(
            lambda t: self.function(t).ravel()
        )

    def reshape(self, *args, **kwargs) -> "ArrayMapper":
        """Numpy reshape

        """
        return ArrayMapper(
            lambda t: self.function(t).reshape(*args, **kwargs)
        )


def lift(f):
    """Lift a function

    lift :: a -> b -> ArrayMapper(a) -> ArrayMapper(b) -> ArrayMapper(b)

    """

    @wraps(f)
    def lifted(mapper):
        return ArrayMapper(compose(f, mapper.function))

    return lifted


x = ArrayMapper()
x.__doc__ = (
    """ArrayMapper instantiated at load-time for convenient importing to applications

    Intended usage is to import ``x`` standalone:

    .. code-block:: python

        import gammy
        from gammy.arraymapper import x

        # Define formula
        formula = gammy.Scalar() * x

    """
)
