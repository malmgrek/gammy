"""Numpy array mapping in model building"""


from gammy.utils import compose


class ArrayMapper():
    """Convenience tool for handling input mappings

    When working with basis function regression, when building the
    design matrix, one needs the information 'how eact term maps input array'.
    This class helps writing human-readable model definitions that explicitly
    show how inputs are mapped. For example,

        ``formula = a(x[:, 0]) * x[:, 1] + x[:, 2] ** 2``

    is a valid definition as long as ``a`` is an instance of
    :class:`gammy.Formula` and ``x`` is an instance of :class:`ArrayMapper`.

    TODO: Are some natural methods are still missing?

    """

    def __init__(self, function=lambda t: t):
        self.function = function
        return

    def __call__(self, *args, **kwargs):
        return self.function.__call__(*args, **kwargs)

    def __getitem__(self, key):
        return ArrayMapper(
            lambda t: t.__getitem__(key)
        )

    def __add__(self, other):
        return ArrayMapper(
            lambda t: self.function(t).__add__(other.function(t))
        )

    def __sub__(self, other):
        return ArrayMapper(
            lambda t: self.function(t).__sub__(other.function(t))
        )

    def __mul__(self, other):
        return ArrayMapper(
            lambda t: self.function(t).__mul__(other.function(t))
        )

    def __truediv__(self, other):
        return ArrayMapper(
            lambda t: self.function(t).__truediv__(other.function(t))
        )

    def __pow__(self, n):
        return ArrayMapper(
            lambda t: self.function(t).__pow__(n)
        )

    def __neg__(self):
        return ArrayMapper(
            lambda t: self.function(t).__neg__()
        )

    def lift(self, f):
        return ArrayMapper(
            compose(f, self.function)
        )

    def ravel(self):
        return ArrayMapper(
            lambda t: self.function(t).ravel()
        )

    def reshape(self, *args, **kwargs):
        return ArrayMapper(
            lambda t: self.function(t).reshape(*args, **kwargs)
        )


x = ArrayMapper()
