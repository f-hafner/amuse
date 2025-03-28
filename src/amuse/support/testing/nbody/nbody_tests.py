from sys import exception
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

particle_inputs_collision_detection = (
        7, {"x": [-101.0, -100.0, -0.5, 0.5, 100.0, 101.0, 104.0] | nbody_system.length,
            "y": 0 | nbody_system.length,
            "z": 0 | nbody_system.length,
            "mass": 0.001 | nbody_system.mass,
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
    instance.parameters.opening_angle = 0.1
    particles = particle_fixture
    instance.particles.add_particles(particles)
    collisions = instance.stopping_conditions.collision_detection
    collisions.enable()
    instance.evolve_model(1.0 | nbody_system.time)

    assert collisions.is_set(), "collisions not set"
    assert instance.model_time < 0.5 | nbody_system.time, "time too big"
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



