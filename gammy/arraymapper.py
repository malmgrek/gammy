"""Arraymapper module

.. autosummary::
   :toctree:

   ArrayMapper

"""

from __future__ import annotations

from gammy.utils import compose


class ArrayMapper():
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
        return

    def __call__(self, *args, **kwargs):
        return self.function.__call__(*args, **kwargs)

    def __getitem__(self, key: int) -> ArrayMapper:
        """Access index as in a NumPy Array

        """
        return ArrayMapper(
            lambda t: t.__getitem__(key)
        )

    def __add__(self, other) -> ArrayMapper:
        """Addition

        """
        return ArrayMapper(
            lambda t: self.function(t).__add__(other.function(t))
        )

    def __sub__(self, other) -> ArrayMapper:
        """Subtraction

        """
        return ArrayMapper(
            lambda t: self.function(t).__sub__(other.function(t))
        )

    def __mul__(self, other) -> ArrayMapper:
        """Multiplication

        """
        return ArrayMapper(
            lambda t: self.function(t).__mul__(other.function(t))
        )

    def __truediv__(self, other) -> ArrayMapper:
        """Division

        """
        return ArrayMapper(
            lambda t: self.function(t).__truediv__(other.function(t))
        )

    def __pow__(self, n: float) -> ArrayMapper:
        """Raise to exponent

        """
        return ArrayMapper(
            lambda t: self.function(t).__pow__(n)
        )

    def __neg__(self) -> ArrayMapper:
        """Negation operation

        """
        return ArrayMapper(
            lambda t: self.function(t).__neg__()
        )

    def lift(self, f):
        """Lift the contained function with a given function

        """
        return ArrayMapper(
            compose(f, self.function)
        )

    def ravel(self) -> ArrayMapper:
        """NumPy ravel method

        """
        return ArrayMapper(
            lambda t: self.function(t).ravel()
        )

    def reshape(self, *args, **kwargs) -> ArrayMapper:
        """Numpy reshape

        """
        return ArrayMapper(
            lambda t: self.function(t).reshape(*args, **kwargs)
        )


x = ArrayMapper()
