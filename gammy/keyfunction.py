
import attr

from gammy.utils import compose


@attr.s
class KeyFunction:

    # TODO: Implement other operations that should be supported

    function = attr.ib(default=lambda t: t)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __getitem__(self, key):
        return KeyFunction(lambda t: t[key])

    def __mul__(self, other):
        return KeyFunction(lambda t: self.function(t) * other.function(t))

    def __pow__(self, n):
        return KeyFunction(function=lambda t: self.function(t) ** n)

    def lift(self, f):
        return KeyFunction(function=compose(f, self.function))


x = KeyFunction()