import pytest

import numpy as np
import time
import math
import pytest

from amuse.community.bhtree.interface import BHTree
from amuse.support.exceptions import AmuseException
from amuse.support.core import OrderedDictionary
from amuse.units import nbody_system
from amuse.units import units
from amuse.units import quantities

from amuse import datamodel
from amuse.ic import plummer
from pytest import fixture
from equality_with_units import assert_equal
from equality_with_units import assert_equal_with_abstol
from equality_with_units import assert_equal_with_reltol

particle_inputs_new_particle = [
    15.0 | nbody_system.mass,
    10.0 | nbody_system.length,
    20.0 | nbody_system.length,
    30.0 | nbody_system.length,
    1.0 | nbody_system.speed,
    1.0 | nbody_system.speed,
    3.0 | nbody_system.speed,
    10.0 | nbody_system.length
    ]

particle_inputs_new_particle_kg = [
    15.0 | units.kg,
    10.0 | units.m,
    20.0 | units.m,
    30.0 | units.m,
    0.0 | units.m/units.s,
    0.0 | units.m/units.s,
    0.0 | units.m/units.s,
    10.0 | units.m
    ]


# Note how bhtree and bhtree_kg fixtures are passed as strings
    # and evaluated with request.getfixturevalue. See also:
    # https://miguendes.me/how-to-use-fixtures-as-arguments-in-pytestmarkparametrize
    # https://github.com/pytest-dev/pytest/issues/349
@pytest.mark.parametrize(
        ("nbody_input", "particle_inputs"),
        [("nbody_instance", particle_inputs_new_particle),
         ("nbody_instance_kg", particle_inputs_new_particle_kg)]
        )
def test_new_particle_generic_version(nbody_input, particle_inputs, request): # combines test4 and test5
    nbody_instance = request.getfixturevalue(nbody_input)
    index = nbody_instance.new_particle(*particle_inputs)
    nbody_instance.commit_particles()
    assert_equal(nbody_instance.get_mass(index), particle_inputs[0])
    assert_equal(nbody_instance.get_radius(index), particle_inputs[1])





