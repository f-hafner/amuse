"""Functions for using pytest with units."""


from amuse.units.quantities import none
from amuse.units.quantities import to_quantity
from amuse.units.quantities import is_quantity
from pytest import approx


def check_comparable(x, y):
    if is_quantity(x):
        if not is_quantity(y) and not x.unit.base == none.base:
            raise TypeError("Cannot compare quantity: {0} with non-quantity: {1}.".format(x, y))
    elif is_quantity(y):
        if not y.unit.base == none.base:
            raise TypeError("Cannot compare non-quantity: {0} with quantity: {1}.".format(x, y))


def convert_to_numeric(x, y, in_units):
    if in_units:
        return (x.value_in(in_units), y.value_in(in_units))
    elif is_quantity(x) or is_quantity(y):
        return (
                to_quantity(x).value_in(to_quantity(y).unit),
                to_quantity(y).value_in(to_quantity(y).unit)
                )
    else:
        return (x, y)

# helper function to check almost equal
# NOTE: this is currently written for a scalar; the original
# functions work with np.arrays
# Also, we'll have to add functionality for the units here
# TODO: add regression tests for new checks and old checks?
def assert_equal_with_abstol(x, y, digits, msg=""):
    """Ported from failUnlessAlmostEqual."""
    check_comparable(x, y)
    x_num, y_num = convert_to_numeric(x, y, in_units=None)
    assert x_num == approx(y_num, abs=10**(-digits)), msg


def assert_equal_with_reltol(x, y, digits=8, msg="", in_units=None):
    """Ported from failUnlessAlmostRelativeEqual."""
    check_comparable(x, y)
    x_num, y_num = convert_to_numeric(x, y, in_units=in_units)
    assert x_num == approx(y_num, rel=10**(-digits)), msg


# TODO: rename the above with units? or find better name in general
def assert_equal_units(x, y, msg="", in_units=None):
    """Ported from failUnlessEqual."""
    check_comparable(x, y)
    x_num, y_num = convert_to_numeric(x, y, in_units)
    assert x_num == y_num, msg

