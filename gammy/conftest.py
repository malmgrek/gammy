"""Top-level configuration for PyTest

"""

import pytest

import gammy


@pytest.fixture(scope="module", params=[
    lambda formula, *args: lambda input_data, y: gammy.bayespy.GAM(
        # HACK: Let's use *args to allow feeding the `tau` parameter
        # only for `gammy.numpy.GAM`. Thus, we leave *args unused for
        # `gammy.bayespy.GAM`.
        formula
    ).fit(input_data, y),
    lambda formula, *args: lambda input_data, y: gammy.numpy.GAM(
        formula, *args
    ).fit(input_data, y)
])
def fit_model(request):
    return request.param
