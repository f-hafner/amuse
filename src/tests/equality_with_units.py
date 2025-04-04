"""Functions for using pytest with units."""
from pytest import approx
import numpy as np
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose
from amuse.units.quantities import none
from amuse.units.quantities import to_quantity
from amuse.units.quantities import is_quantity

PRECISION = int(round(np.log10(2.0/(np.finfo(np.double).eps))))-1

def _check_comparable(x, y):
    if is_quantity(x):
        if not is_quantity(y) and not x.unit.base == none.base:
            raise TypeError("Cannot compare quantity: {0} with non-quantity: {1}.".format(x, y))
    elif is_quantity(y):
        if not y.unit.base == none.base:
            raise TypeError("Cannot compare non-quantity: {0} with quantity: {1}.".format(x, y))


def _convert_to_numeric(x, y, in_units):
    if in_units:
        return (x.value_in(in_units), y.value_in(in_units))
    elif is_quantity(x) or is_quantity(y):
        return (
                to_quantity(x).value_in(to_quantity(y).unit),
                to_quantity(y).value_in(to_quantity(y).unit)
                )
    else:
        return (x, y)


def assert_equal_with_abstol(x, y, digits, msg=""):
    """Ported from failUnlessAlmostEqual."""
    _check_comparable(x, y)
    x_num, y_num = _convert_to_numeric(x, y, in_units=None)
    assert_allclose(x_num, y_num, atol=10**(-digits), err_msg=msg)


def assert_equal_with_reltol(x, y, digits=PRECISION, msg="", in_units=None):
    """Ported from failUnlessAlmostRelativeEqual."""
    _check_comparable(x, y)
    x_num, y_num = _convert_to_numeric(x, y, in_units=in_units)
    assert_allclose(x_num, y_num, rtol=10**(-digits), err_msg=msg)


def assert_equal(x, y, msg="", in_units=None):
    """Ported from failUnlessEqual."""
    _check_comparable(x, y)
    x_num, y_num = _convert_to_numeric(x, y, in_units)
    assert_array_equal(x_num, y_num, err_msg=msg)

