from sys import exception
import pytest

import numpy as np
import time
import math
import pytest
import logging

from amuse.community.bhtree.interface import BHTree
from amuse.support.exceptions import AmuseException, CoreException
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


logger = logging.getLogger(__name__)

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

particle_inputs_collision_detection = (
        7, {"x": [-101.0, -100.0, -0.5, 0.5, 100.0, 101.0, 104.0] | nbody_system.length,
            "y": 0 | nbody_system.length,
            "z": 0 | nbody_system.length,
            "mass": 0.001 | nbody_system.mass, # TODO: differs ph4 vs bhtree
            "radius": 0.01 | nbody_system.length,
            "velocity": [[2, 0, 0], [-2, 0, 0]]*3 + [[-4, 0, 0]] | nbody_system.speed
          })

# Note how nbody_instance and nbody_instance_kg fixtures are passed as strings
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



particle_inputs_gravity_with_same_potential = (
        2, {"mass": [1.0, 1.0] | nbody_system.mass,
            "radius": [0.0001, 0.0001] | nbody_system.length,
            "position": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]] | nbody_system.length,
            "velocity": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] | nbody_system.speed
          })

particle_inputs_gravity_at_positions = (
        6, {"mass": 1.0 | nbody_system.mass,
            "radius": 0.0001 | nbody_system.length,
            "position": [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]] | nbody_system.length,
            "velocity": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] | nbody_system.speed
          })

# NOTE: 0.125 is the default value for epsilon_squared;
# it is put here to make things explicit.
@pytest.mark.parametrize(
    ("particle_fixture", "point", "epsilon2"),
    [(particle_inputs_gravity_with_same_potential, 1.0, 0.00001),
     (particle_inputs_gravity_at_positions, 0.0, 0.125)],
    indirect=["particle_fixture"]
)
def test_zero_gravity_generic_version(nbody_instance, particle_fixture, point, epsilon2):
    """Test gravity at point where it is expected to be 0, depending on parameters."""
    nbody_instance.parameters.epsilon_squared = epsilon2 | nbody_system.length**2
    nbody_instance.particles.add_particles(particle_fixture)

    zero = 0.0 | nbody_system.length
    point = point | nbody_system.length
    gravity = nbody_instance.get_gravity_at_point(zero, point, zero, zero)
    for f in gravity:
        assert_equal_with_reltol(f, 0.0 | nbody_system.acceleration, 3)



@pytest.mark.parametrize(
    "particle_fixture",
    [particle_inputs_collision_detection],
    indirect=True
)
def test_collision_detection_generic_version(make_nbody_instance, particle_fixture): # formerly test17
    instance = make_nbody_instance(redirection="none")
    instance.initialize_code()
    instance.parameters.set_defaults()
    try:
        instance.parameters.opening_angle = 0.1
    except CoreException: # necessary for ph4
        logger.warning("Skipped setting opening_angle: no such parameter.")

    particles = particle_fixture
    instance.particles.add_particles(particles)
    collisions = instance.stopping_conditions.collision_detection
    collisions.enable()
    instance.evolve_model(1.0 | nbody_system.time)

    assert collisions.is_set(), "collisions not set"
    assert instance.model_time < 0.5 | nbody_system.time, "time too big"
    # TODO: "PH4 can handle only one collision at a time", test_ph4.py, line 464
    assert len(collisions.particles(0)) == 3, "mismatch in N particles"
    assert len(collisions.particles(1)) == 3, "mismatch in N particles"
    assert len(particles - collisions.particles(0) - collisions.particles(1)) == 1

    left = abs(collisions.particles(0).x - collisions.particles(1).x)
    right = collisions.particles(0).radius + collisions.particles(1).radius
    assert all(left < right) # TODO: useful error message?

    sticky_merged = datamodel.Particles(len(collisions.particles(0)))
    sticky_merged.mass = collisions.particles(0).mass + collisions.particles(1).mass
    sticky_merged.radius = collisions.particles(0).radius
    for p1, p2, merged in zip(collisions.particles(0), collisions.particles(1), sticky_merged):
        merged.position = (p1 + p2).center_of_mass()
        merged.velocity = (p1 + p2).center_of_mass_velocity()

    instance.particles.remove_particles(collisions.particles(0) + collisions.particles(1))
    instance.particles.add_particles(sticky_merged)

    instance.evolve_model(1.0 | nbody_system.time)

    assert collisions.is_set(), "collisions not set"
    assert instance.model_time < 1.0 | nbody_system.time, "time too big"
    assert len(collisions.particles(0)) == 1, "mismatch in N particles"
    assert len(collisions.particles(1)) == 1, "mismatch in N particles"
    assert len(instance.particles - collisions.particles(0) - collisions.particles(1)) == 2

    left = abs(collisions.particles(0).x - collisions.particles(1).x)
    right = collisions.particles(0).radius + collisions.particles(1).radius
    assert all(left < right) # TODO: useful error message?


def test_system_sun_earth_generic_version(make_nbody_instance):
    # Set up the instance
    convert_nbody = nbody_system.nbody_to_si(1.0 | units.MSun, 149.5e6 | units.km) # for test1
    instance = make_nbody_instance(convert_nbody)
    instance.parameters.epsilon_squared = 0.001 | units.AU**2
    instance.commit_parameters()

    stars = datamodel.Stars(2)

    sun = stars[0]
    sun.mass = 1.0 | units.Msun
    sun.position = [0.0, 0.0, 0.0] | units.m
    sun.velocity = [0.0, 0.0, 0.0] | units.ms
    sun.radius = 1.0 | units.Rsun

    earth = stars[1]
    earth.mass = 5.9736e24 | units.kg
    earth.radius = 6371 | units.km
    earth.position = [149.5e6, 0.0, 0.0] | units.km
    earth.velocity = [0.0, 29800, 0.0] | units.ms

    instance.particles.add_particles(stars)

    # Tests
    position_at_start = earth.position.value_in(units.AU)[0]

    instance.evolve_model(365.0 | units.day)
    instance.particles.copy_values_of_all_attributes_to(stars)

    position_after_full_rotation = earth.position.value_in(units.AU)[0]
    #assert_equal_with_abstol(position_at_start, position_after_full_rotation, 3)
    # Replaced with
    assert_equal_with_abstol(position_at_start, position_after_full_rotation, 2)

    instance.evolve_model(365.0 + (365.0 / 2) | units.day)
    instance.particles.copy_values_of_all_attributes_to(stars)
    position_after_half_a_rotation = earth.position.value_in(units.AU)[0]


    position_delta = position_at_start + position_after_half_a_rotation
    #assert_equal_with_abstol(-position_at_start, position_after_half_a_rotation, 2)
    # Replace with:
    assert_equal_with_abstol(position_delta, 0.00, 2)

    instance.evolve_model(365.0 + (365.0 / 2) + (365.0 / 4) | units.day)
    instance.particles.copy_values_of_all_attributes_to(stars)
    position_after_half_a_rotation = earth.position.value_in(units.AU)[1]

    assert_equal_with_abstol(-position_at_start, position_after_half_a_rotation, 1)


# taken from bhtree tests, test_total_energy
def test_energy_unchanged_generic_version(make_nbody_instance):
    # Setup
    np.random.seed(0)
    number_of_stars = 2
    stars = plummer.new_plummer_model(number_of_stars)
    stars.radius = 0.00001 | nbody_system.length
    stars.scale_to_standard()

    instance = make_nbody_instance()
    instance.initialize_code()
    instance.parameters.epsilon_squared = (1.0 / 20.0 / (number_of_stars**0.33333) | nbody_system.length)**2
    if hasattr(instance.parameters, "timestep_parameter"): # ph4
        instance.parameters.timestep_parameter = 0.01
    elif hasattr(instance.parameters, "timestep"): # bhtree
        # original code (test16 in tests of bhtree) had 2 lines, one with
        # 0.004 and one with 0.00001
        instance.parameters.timestep = 0.00001 | nbody_system.time
    else:
        msg = "No parameter for timesteps found."
        raise AttributeError(msg)

    instance.commit_parameters()
    instance.particles.add_particles(stars)
    instance.commit_particles()

    # Test
    energy_total_t0 = instance.potential_energy + instance.kinetic_energy
    request = instance.evolve_model.asynchronous(1.0 | nbody_system.time)
    request.result()
    energy_total_t1 = instance.potential_energy + instance.kinetic_energy

    assert_equal_with_reltol(energy_total_t0, energy_total_t1, 3)

# TODO: rename the function
@pytest.mark.parametrize("n_workers", [1, 4])
def test_energy_changed_generic_version(make_nbody_instance, n_workers): #tests10a/b from ph4
    # Setup
    instance = make_nbody_instance(number_of_workers=n_workers)
    instance.initialize_code()

    instance.parameters.epsilon_squared = 0.0 | nbody_system.length**2
    if hasattr(instance.parameters, "timestep_parameter"): # ph4
        instance.parameters.timestep_parameter = 0.01
    elif hasattr(instance.parameters, "timestep"): # bhtree
        # original code (test16 in tests of bhtree) had 2 lines, one with
        # 0.004 and one with 0.00001
        instance.parameters.timestep = 0.00001 | nbody_system.time
    else:
        msg = "No parameter for timesteps found."
        raise AttributeError(msg)
    instance.commit_parameters()

    number_of_stars = 100
    stars = plummer.new_plummer_model(number_of_stars)
    instance.particles.add_particles(stars)
    channel = stars.new_channel_to(instance.particles)


    # Test
    instance.evolve_model(0.001 | nbody_system.time)
    e0 = instance.kinetic_energy + instance.potential_energy
    stars.mass *= 0.9
    channel.copy()

    instance.synchronize_model()
    e1 = instance.kinetic_energy + instance.potential_energy

    assert e1 != e0


