import attr

from gammy.utils import compose


@attr.s
class ArrayMapper():

    # TODO: Implement other operations that should be supported
    # TODO: Test operations

    function = attr.ib(default=lambda t: t)

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

    def __div__(self, other):
        return ArrayMapper(
            lambda t: self.function(t).__div__(other.function(t))
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
