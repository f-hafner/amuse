
# -*- coding: utf-8 -*-
import time
import pytest

from amuse.support.core import OrderedDictionary
from amuse.units import nbody_system
from amuse.units import units

from amuse.support.testing.equality_with_units import assert_equal_with_reltol

## Test whether standard nbody-code interfaces are implemented correctly
from amuse.support.testing.nbody.nbody_tests import *

### Bhtree-specific tests
def test_bhtree_parameters(make_nbody_instance):
    # Setup
    convert_nbody = nbody_system.nbody_to_si(1.0 | units.yr, 1.0 | units.AU)
    instance = make_nbody_instance(convert_nbody)
    instance.commit_parameters()

    eps2 = instance.legacy_interface.get_epsilon_squared()
    assert isinstance(eps2, OrderedDictionary)
    assert eps2.keys() == ["epsilon_squared", "__result"]
    assert eps2.values() == [0.125, 0]
    assert_equal_with_reltol(instance.parameters.epsilon_squared, 0.125 | units.AU**2, in_units=units.AU**2)

    for x in [0.01, 0.1, 0.2]:
        instance.parameters.epsilon_squared = x | units.AU**2
        assert_equal_with_reltol(instance.parameters.epsilon_squared, x | units.AU**2, in_units=units.AU**2)

    time_step = instance.legacy_interface.get_time_step()
    assert isinstance(time_step, OrderedDictionary)
    assert time_step.keys() == ["time_step", "__result"]
    assert time_step.values() == [0.015625, 0]

    assert_equal_with_reltol(instance.parameters.timestep, 0.015625 | units.yr, in_units=units.yr)

    for x in [0.001, 0.01, 0.1]:
        instance.parameters.timestep = x | units.yr
        assert_equal_with_reltol(instance.parameters.timestep, x | units.yr, in_units=units.yr)

    theta = instance.legacy_interface.get_theta_for_tree()
    assert isinstance(theta, OrderedDictionary)
    assert theta.keys() == ["theta_for_tree", "__result"]
    assert theta.values() == [0.75, 0]

    assert instance.parameters.opening_angle == 0.75
    for x in [0.2, 0.5, 0.7]:
        instance.parameters.opening_angle = x
        assert instance.parameters.opening_angle == x, "does not change opening angle parameter"

    gravity = instance.legacy_interface.get_use_self_gravity()
    assert isinstance(gravity, OrderedDictionary)
    assert gravity.keys() == ["use_self_gravity", "__result"]
    assert gravity.values() == [1, 0]

    assert instance.parameters.use_self_gravity == 1
    for x in [0, 1]:
        instance.parameters.use_self_gravity = x
        assert instance.parameters.use_self_gravity == x, "does not change use_self_gravity parameter"

    ncrit = instance.legacy_interface.get_ncrit_for_tree()
    assert isinstance(ncrit, OrderedDictionary)
    assert ncrit.keys() == ["ncrit_for_tree", "__result"]
    assert ncrit.values() == [12, 0]

    assert instance.parameters.ncrit_for_tree == 12
    for x in [512, 2048, 4096]:
        instance.parameters.ncrit_for_tree = x
        assert instance.parameters.ncrit_for_tree == x, "does not change ncrit_for_tree parameter"

    dt_dia = instance.legacy_interface.get_dt_dia()
    assert isinstance(dt_dia, OrderedDictionary)
    assert dt_dia.keys() == ["dt_dia", "__result"]
    assert dt_dia.values() == [1.0, 0]

    assert_equal_with_reltol(instance.parameters.dt_dia, 1.0 | units.yr, in_units=units.yr)
    for x in [0.1, 10.0, 100.0]:
        instance.parameters.dt_dia = x | units.yr
        assert_equal_with_reltol(instance.parameters.dt_dia, x | units.yr, in_units=units.yr)


@pytest.mark.parametrize(
    "particle_fixture",
    [particle_inputs_stop_n_steps],
    indirect=True
)
def test_stop_n_steps(nbody_instance, particle_fixture):
    nbody_instance.particles.add_particles(particle_fixture)
    nbody_instance.stopping_conditions.number_of_steps_detection.enable()
    nbody_instance.parameters.stopping_conditions_number_of_steps = 2
    assert nbody_instance.parameters.stopping_conditions_number_of_steps == 2

    nbody_instance.evolve_model(10 | nbody_system.time)
    assert nbody_instance.stopping_conditions.number_of_steps_detection.is_set()
    assert nbody_instance.model_time < 10 | nbody_system.time



@pytest.mark.parametrize(
    "particle_fixture",
    [particle_inputs_stop_n_steps],
    indirect=True
)
def test_stop_timeout(nbody_instance, particle_fixture):
    nbody_instance.particles.add_particles(particle_fixture)

    nbody_instance.stopping_conditions.timeout_detection.enable()

    very_short_time_to_evolve = 1 | units.s
    very_long_time_to_evolve = 1e9 | nbody_system.time

    nbody_instance.parameters.stopping_conditions_timeout = very_short_time_to_evolve

    assert nbody_instance.parameters.stopping_conditions_timeout == very_short_time_to_evolve

    start = time.time()
    nbody_instance.evolve_model(very_long_time_to_evolve)
    end = time.time()

    assert nbody_instance.stopping_conditions.timeout_detection.is_set()
    assert very_short_time_to_evolve.value_in(units.s) + 2 > end - start, "early stopping fails"

    nbody_instance.cleanup_code()

