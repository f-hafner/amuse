
# -*- coding: utf-8 -*-
import numpy
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

from amusetest_helpers import assert_equal
from amusetest_helpers import assert_equal_with_abstol
from amusetest_helpers import assert_equal_with_reltol

particle_inputs_kg = (2, {"mass": [15.0, 30.0] | units.kg,
          "radius": [10.0, 20.0] | units.m,
          "position": [[10.0, 20.0, 30.0], [20.0, 40.0, 60.0]] | units.m,
          "velocity": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] | units.m / units.s
          })
particle_inputs_test13 = (2, {"mass": [30.0, 30.0] | units.kg,
          "radius": [1.0, 1.0] | units.m,
          "position": [[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]] | units.m,
          "velocity": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] | units.m / units.s
          })

particle_inputs_test18 = (
        2, {"x": [0.0, 10.0] | nbody_system.length,
            "y": 0 | nbody_system.length,
            "z": 0 | nbody_system.length,
            "radius": 0.005 | nbody_system.length,
            "vx": 0 | nbody_system.speed,
            "vy": 0 | nbody_system.speed,
            "vz": 0 | nbody_system.speed,
            "mass": 1.0 | nbody_system.mass,
          })

particle_inputs_test23 = (
        2, {"x": [0.0, 10.0] | nbody_system.length,
            "y": 0.0 | nbody_system.length,
            "z": 0.0 | nbody_system.length,
            "vx": 1.0 | nbody_system.speed,
            "vy": 0.0 | nbody_system.speed,
            "vz": 0.0 | nbody_system.speed,
            "mass": 0.1 | nbody_system.mass,
          })

particle_inputs_test9 = (
        2, {"mass": [1.0, 1.0] | nbody_system.mass,
            "radius": [0.0001, 0.0001] | nbody_system.length,
            "position": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]] | nbody_system.length,
            "velocity": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] | nbody_system.speed
          })
particle_inputs_test10 = (
        6, {"mass": 1.0 | nbody_system.mass,
            "radius": 0.0001 | nbody_system.length,
            "position": [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]] | nbody_system.length,
            "velocity": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] | nbody_system.speed
          })
particle_inputs_collision_detection = (
        7, {"x": [-101.0, -100.0, -0.5, 0.5, 100.0, 101.0, 104.0] | nbody_system.length,
            "y": 0 | nbody_system.length,
            "z": 0 | nbody_system.length,
            "mass": 0.001 | nbody_system.mass,
            "radius": 0.01 | nbody_system.length,
            "velocity": [[2, 0, 0], [-2, 0, 0]]*3 + [[-4, 0, 0]] | nbody_system.speed
          })


# Factory to create bhtrees. Handles teardown for all
# Follows https://docs.pytest.org/en/stable/how-to/fixtures.html#factories-as-fixtures
@fixture()
def make_bhtree():
    created_bhtrees = []

    def _make_bhtree(x=None, **kwargs):
        if not x:
            tree = BHTree(**kwargs)
        else:
            tree = BHTree(x, **kwargs)
        created_bhtrees.append(tree)
        return tree

    yield _make_bhtree

    for tree in created_bhtrees:
        tree.cleanup_code()
        tree.stop()


@fixture
def particle_fixture(request):
    num_particles, kwargs = request.param
    particles = datamodel.Particles(num_particles)
    for key, value in kwargs.items():
        setattr(particles, key, value)
    return particles



@fixture
def bhtree_msun(make_bhtree):
    convert_nbody = nbody_system.nbody_to_si(1.0 | units.MSun, 149.5e6 | units.km) # for test1
    instance = make_bhtree(convert_nbody)
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
    #instance.commit_particles() # this disables adding particles later on
    yield stars, instance


@fixture
def bhtree_kg(make_bhtree): # for test4, test6, test7, test11, test12, test13
    convert_nbody = nbody_system.nbody_to_si(5.0 | units.kg, 10.0 | units.m)
    instance = make_bhtree(convert_nbody)
    instance.commit_parameters()
    yield instance


@fixture
def bhtree_test14(make_bhtree):
    convert_nbody = nbody_system.nbody_to_si(1.0 | units.yr, 1.0 | units.AU)
    instance = make_bhtree(convert_nbody)
    instance.commit_parameters()
    yield instance


@fixture
def bhtree_for_epsilon_squared_test(make_bhtree):
    convert_nbody = nbody_system.nbody_to_si(1.0 | units.MSun, 1.0 | units.AU)

    particles = datamodel.Particles(2)
    sun = particles[0]
    sun.mass = 1.0 | units.MSun
    sun.position = [0.0, 0.0, 0.0] | units.AU
    sun.velocity = [0.0, 0.0, 0.0] | units.AU / units.yr
    sun.radius = 1.0 | units.RSun

    earth = particles[1]
    earth.mass = 5.9736e24 | units.kg
    earth.radius = 6371.0 | units.km
    earth.position = [0.0, 1.0, 0.0] | units.AU
    earth.velocity = [2.0*numpy.pi, -0.0001, 0.0] | units.AU / units.yr

    initial_direction = math.atan((earth.velocity[0]/earth.velocity[1]))
    final_direction = []
    for log_eps2 in range(-9, 10, 2): # TODO: use parameterize here? - but it's a fixture
        instance = make_bhtree(convert_nbody)
        instance.initialize_code()
        instance.parameters.epsilon_squared = 10.0**log_eps2 | units.AU ** 2
        instance.particles.add_particles(particles)
        instance.commit_particles()
        instance.evolve_model(0.25 | units.yr)
        final_direction.append(math.atan((instance.particles[1].velocity[0] /
            instance.particles[1].velocity[1])))

    yield instance, initial_direction, final_direction


@fixture
def bhtree_to_test_energy(make_bhtree):
    numpy.random.seed(0)
    number_of_stars = 2
    stars = plummer.new_plummer_model(number_of_stars)
    stars.radius = 0.00001 | nbody_system.length
    stars.scale_to_standard()

    instance = make_bhtree()
    instance.initialize_code()
    instance.parameters.epsilon_squared = (1.0 / 20.0 / (number_of_stars**0.33333) | nbody_system.length)**2
    instance.parameters.timestep = 0.004 | nbody_system.time
    instance.parameters.timestep = 0.00001 | nbody_system.time
    instance.commit_parameters()
    instance.particles.add_particles(stars)
    instance.commit_particles()

    yield instance



def test_test1(bhtree_msun):
    stars, bhtree = bhtree_msun
    sun, earth = stars
    postion_at_start = earth.position.value_in(units.AU)[0]

    bhtree.evolve_model(365.0 | units.day)
    bhtree.particles.copy_values_of_all_attributes_to(stars)

    postion_after_full_rotation = earth.position.value_in(units.AU)[0]
    assert_equal_with_abstol(postion_at_start, postion_after_full_rotation, 3)

    bhtree.evolve_model(365.0 + (365.0 / 2) | units.day)
    bhtree.particles.copy_values_of_all_attributes_to(stars)
    postion_after_half_a_rotation = earth.position.value_in(units.AU)[0]

    assert_equal_with_abstol(-postion_at_start, postion_after_half_a_rotation, 2)

    bhtree.evolve_model(365.0 + (365.0 / 2) + (365.0 / 4) | units.day)
    bhtree.particles.copy_values_of_all_attributes_to(stars)
    postion_after_half_a_rotation = earth.position.value_in(units.AU)[1]

    assert_equal_with_abstol(-postion_at_start, postion_after_half_a_rotation, 1)


def test_test4(bhtree_kg):
    bhtree = bhtree_kg

    index = bhtree.new_particle(
        15.0 | units.kg,
        10.0 | units.m, 20.0 | units.m, 30.0 | units.m,
        0.0 | units.m/units.s, 0.0 | units.m/units.s, 0.0 | units.m/units.s,
        10.0 | units.m
    )
    bhtree.commit_particles()
    assert_equal(bhtree.get_mass(index), 15.0 | units.kg, "new particle not added correctly")
    assert_equal(bhtree.get_radius(index), 10.0 | units.m)


def test_test5(make_bhtree):
    instance = make_bhtree()

    index = instance.new_particle(
        15.0 | nbody_system.mass,
        10.0 | nbody_system.length, 20.0 | nbody_system.length, 30.0 | nbody_system.length,
        1.0 | nbody_system.speed, 1.0 | nbody_system.speed, 3.0 | nbody_system.speed,
        10.0 | nbody_system.length
    )
    instance.commit_particles()
    assert_equal(instance.get_radius(index), 10.0 | nbody_system.length)
    assert_equal(instance.get_mass(index), 15.0 | nbody_system.mass)



def test_test6(bhtree_kg):
    instance = bhtree_kg
    indices = instance.new_particle(
        [15.0, 30.0] | units.kg,
        [10.0, 20.0] | units.m, [20.0, 40.0] | units.m, [30.0, 50.0] | units.m,
        [0.0, 0.01] | units.m/units.s, [0.0, 0.01] | units.m/units.s, [0.0, 0.01] | units.m/units.s,
        [10.0, 20.0] | units.m
    )
    instance.commit_particles()

    assert_equal(instance.get_mass(indices[0]), 15.0 | units.kg)

    with pytest.raises(AmuseException) as excinfo:
        instance.get_mass([4, 5])
        assert "Error when calling 'get_mass' of a 'BHTree', errorcode is -1" in str(excinfo.value)


@pytest.mark.parametrize(
    "particle_fixture, raw_particle_data",
    [(particle_inputs_kg, particle_inputs_kg)],
    indirect=["particle_fixture"]
)
def test_test7(bhtree_kg, particle_fixture, raw_particle_data):
    instance = bhtree_kg
    instance.particles.add_particles(particle_fixture)
    instance.commit_particles()

    expected_count = raw_particle_data[0]
    assert len(instance.particles) == expected_count

    expected_mass = raw_particle_data[1]["mass"]
    for idx in range(expected_count):
        assert_equal(instance.get_mass(idx+1), expected_mass[idx])

    expected_radius = raw_particle_data[1]["radius"]
    for idx in range(expected_count):
        assert_equal(instance.get_radius(idx+1), expected_radius[idx])

    expected_position = raw_particle_data[1]["position"]
    for idx in range(expected_count):
        pos = instance.get_position(idx+1)
        pos_e = expected_position[idx]
        for x, x_e in zip(pos, pos_e):
            assert_equal_with_reltol(x, x_e)

    expected_velocity = raw_particle_data[1]["velocity"]
    for idx in range(expected_count):
        vel = instance.get_velocity(idx+1)
        vel_e = expected_velocity[idx]
        for x, x_e in zip(vel, vel_e):
            assert_equal_with_reltol(x, x_e)


@pytest.mark.parametrize(
    "particle_fixture",
    [particle_inputs_kg],
    indirect=True
)
def test_test8(bhtree_kg, particle_fixture):
    instance = bhtree_kg

    instance.particles.add_particles(particle_fixture)
    instance.commit_particles()

    instance.particles.mass = [17.0, 33.0] | units.kg
    assert_equal(instance.get_mass(1), 17.0 | units.kg)


@pytest.mark.parametrize(
    "particle_fixture",
    [particle_inputs_test9],
    indirect=True
)
def test_test9(make_bhtree, particle_fixture):
    instance = make_bhtree()
    instance.parameters.epsilon_squared = 0.00001 | nbody_system.length**2
    instance.particles.add_particles(particle_fixture)

    zero = 0.0 | nbody_system.length
    gravity = instance.get_gravity_at_point(zero, 1.0 | nbody_system.length, zero, zero)
    for f in gravity:
        assert_equal_with_reltol(f, 0.0 | nbody_system.acceleration, 3)

    for x in (0.25, 0.5, 0.75): # TODO: this we should put into a pytest parameterize
        x0 = x | nbody_system.length
        x1 = (2.0 - x) | nbody_system.length
        potential0 = instance.get_potential_at_point(zero, x0, zero, zero)
        potential1 = instance.get_potential_at_point(zero, x1, zero, zero)
        fx0, fy0, fz0 = instance.get_gravity_at_point(zero, x0, zero, zero)
        fx1, fy1, fz1 = instance.get_gravity_at_point(zero, x1, zero, zero)

        assert_equal_with_reltol(fy0, 0.0 | nbody_system.acceleration, 3)
        assert_equal_with_reltol(fz0, 0.0 | nbody_system.acceleration, 3)
        assert_equal_with_reltol(fy1, 0.0 | nbody_system.acceleration, 3)
        assert_equal_with_reltol(fz1, 0.0 | nbody_system.acceleration, 3)

        assert_equal_with_reltol(fx0, -1.0 * fx1, 5)
        fx = (-1.0 / (x0**2) + 1.0 / (x1**2)) * (1.0 | nbody_system.length ** 3 / nbody_system.time ** 2)
        assert_equal_with_reltol(fx, fx0, 2)
        assert_equal_with_reltol(potential0, potential1, 5)


@pytest.mark.parametrize(
    "particle_fixture",
    [particle_inputs_test10],
    indirect=True
)
def test_test10(make_bhtree, particle_fixture):
    instance = make_bhtree()
    instance.particles.add_particles(particle_fixture)
    instance.commit_particles() # TODO: not strictly necessary

    zero = 0.0 | nbody_system.length
    gravity = instance.get_gravity_at_point(zero, zero, zero, zero)
    for f in gravity: # TODO: tested above as well
        assert_equal_with_reltol(f, 0.0 | nbody_system.acceleration, 3)

    for position in (0.25, 0.5, 0.75):
        p0 = position | nbody_system.length
        p1 = -position | nbody_system.length
        for i in range(3):
            args0 = [zero] * 4
            args1 = [zero] * 4
            args0[1 + i] = p0
            args1[1 + i] = p1
            f0 = instance.get_gravity_at_point(*args0)
            f1 = instance.get_gravity_at_point(*args1)

            for j in range(3):
                if j != i:
                    assert_equal_with_reltol(f0[j], 0.0 | nbody_system.acceleration, 3)
                    assert_equal_with_reltol(f1[j], 0.0 | nbody_system.acceleration, 3)
                else:
                    assert_equal_with_reltol(f0[j], -1.0 * f1[j], 5)



@pytest.mark.parametrize(
    "particle_fixture, raw_particle_data",
    [(particle_inputs_kg, particle_inputs_kg)],
    indirect=["particle_fixture"]
)
def test_test11(bhtree_kg, particle_fixture, raw_particle_data):
    instance = bhtree_kg
    instance.particles.add_particles(particle_fixture)

    copyof = instance.particles.copy()
    id_to_check = 1
    expected_mass = raw_particle_data[1]["mass"][id_to_check]
    assert_equal_with_reltol(copyof[id_to_check].mass, expected_mass, 6)

    copyof[1].mass = 35 | units.kg
    copyof.copy_values_of_all_attributes_to(instance.particles)

    assert_equal_with_reltol(instance.particles[1].mass, 35 | units.kg, 6)


@pytest.mark.parametrize(
    "particle_fixture",
    [particle_inputs_kg],
    indirect=True
)
def test_test12(bhtree_kg, particle_fixture):
    instance = bhtree_kg

    instance.particles.add_particles(particle_fixture)
    instance.commit_particles()

    state_values = [16 | units.kg,
             20.0 | units.m,
             40.0 | units.m,
             60.0 | units.m,
             1.0 | units.ms,
             1.0 | units.ms,
             1.0 | units.ms]
    instance.set_state(1, *state_values)
    expected_values = state_values + [0 | units.m]

    curr_state = instance.get_state(1)
    for expected, actual in zip(expected_values, curr_state):
        assert_equal_with_reltol(actual, expected)

    # NOTE: dropped duplicated code in original test12



@pytest.mark.parametrize(
    "particle_fixture",
    [particle_inputs_test13],
    indirect=True
)
def test_test13(bhtree_kg, particle_fixture):
    instance = bhtree_kg

    instance.particles.add_particles(particle_fixture)
    instance.commit_particles()

    com = instance.center_of_mass_position
    assert_equal_with_reltol(com[0], quantities.new_quantity(0.0, units.m))


def test_bhtree_parameters(bhtree_test14):
    instance = bhtree_test14

    # TODO: combine the testing of legacy interface and instance.parameters? or parameterize?
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

def test_effect_of_bhtree_param_epsilon_squared(bhtree_for_epsilon_squared_test):
    instance, initial_direction, final_direction = bhtree_for_epsilon_squared_test

    # Small values of epsilon_squared should result in normal earth-sun dynamics: rotation of 90 degrees
    assert_equal_with_abstol(abs(final_direction[0]), abs(initial_direction + math.pi/2.0), 2)

    # Large values of epsilon_squared should result in ~ no interaction
    assert_equal_with_abstol(final_direction[-1], initial_direction, 2)

    # Outcome is most sensitive to epsilon_squared when epsilon_squared = d(earth, sun)^2
    delta = [abs(final_direction[i+1]-final_direction[i]) for i in range(len(final_direction)-1)]
    assert max(delta) == delta[len(final_direction)//2 - 1]

def test_total_energy(bhtree_to_test_energy): # formerly test16
    instance = bhtree_to_test_energy

    energy_total_t0 = instance.potential_energy + instance.kinetic_energy
    request = instance.evolve_model.asynchronous(1.0 | nbody_system.time)
    request.result()
    energy_total_t1 = instance.potential_energy + instance.kinetic_energy

    assert_equal_with_reltol(energy_total_t0, energy_total_t1, 3)



@pytest.mark.parametrize(
    "particle_fixture",
    [particle_inputs_collision_detection ],
    indirect=True
)
def test_collision_detection(make_bhtree, particle_fixture): # formerly test17
    instance = make_bhtree(redirection="none")
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

    # TODO: this is doing some updates here - put code elsewhere?
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


# TODO: what is the additional benefit of this test? the evolution of the model
# is also tested elsewhere I think?


@pytest.mark.parametrize(
    "particle_fixture",
    [particle_inputs_test18],
    indirect=True
)
def test_test18(make_bhtree, particle_fixture):
    instance = make_bhtree()
    #instance = bhtree_test18
    instance.particles.add_particles(particle_fixture)
    instance.stopping_conditions.number_of_steps_detection.enable()
    instance.parameters.stopping_conditions_number_of_steps = 2
    assert instance.parameters.stopping_conditions_number_of_steps == 2

    instance.evolve_model(10 | nbody_system.time)
    assert instance.stopping_conditions.number_of_steps_detection.is_set()
    assert instance.model_time < 10 | nbody_system.time



@pytest.mark.parametrize(
    "particle_fixture",
    [particle_inputs_test18],
    indirect=True
)
def test_test19(make_bhtree, particle_fixture):
    instance = make_bhtree()
    instance.particles.add_particles(particle_fixture)

    instance.stopping_conditions.timeout_detection.enable()

    very_short_time_to_evolve = 1 | units.s
    very_long_time_to_evolve = 1e9 | nbody_system.time

    instance.parameters.stopping_conditions_timeout = very_short_time_to_evolve

    assert instance.parameters.stopping_conditions_timeout == very_short_time_to_evolve

    start = time.time()
    instance.evolve_model(very_long_time_to_evolve)
    end = time.time()

    assert instance.stopping_conditions.timeout_detection.is_set()
    assert very_short_time_to_evolve.value_in(units.s) + 2 > end - start, "early stopping fails"

    instance.cleanup_code()


@pytest.mark.parametrize(
    "particle_fixture",
    [particle_inputs_test18],
    indirect=True
)
def test_test20(make_bhtree, particle_fixture):
    instance = make_bhtree()
    instance.particles.add_particles(particle_fixture)

    very_short_time_to_evolve = 1 | units.s

    instance.parameters.stopping_conditions_timeout = very_short_time_to_evolve
    assert instance.parameters.stopping_conditions_timeout == very_short_time_to_evolve

    codeparticles1 = instance.particles
    instance.particles.add_particle(datamodel.Particle(
        position=[0, 1, 2] | nbody_system.length,
        velocity=[0, 0, 0] | nbody_system.speed,
        radius=0.005 | nbody_system.length,
        mass=1 | nbody_system.mass
    ))
    codeparticles2 = instance.particles
    assert codeparticles2 is codeparticles1
    instance.cleanup_code()
    codeparticles3 = instance.particles
    assert codeparticles1 is not codeparticles3, "clean up does not work"



@pytest.mark.parametrize(
    "particle_fixture",
    [particle_inputs_test18],
    indirect=True
)
def test_test21(make_bhtree, particle_fixture):
    instance = make_bhtree()
    instance.particles.add_particles(particle_fixture)

    instance.parameters.epsilon_squared = (1e-5 | nbody_system.length)**2
    assert_equal_with_reltol(instance.potential_energy, -0.1 | nbody_system.energy, 5)


@pytest.mark.parametrize(
    "particle_fixture",
    [particle_inputs_test18],
    indirect=True
)
def test_test22(make_bhtree, particle_fixture):
    instance = make_bhtree()
    instance.particles.add_particles(particle_fixture)

    instance.commit_particles()
    p = datamodel.Particle(
        x=1.0 | nbody_system.length,
        y=2.0 | nbody_system.length,
        z=3.0 | nbody_system.length,
        vx=1.0 | nbody_system.speed,
        vy=2.0 | nbody_system.speed,
        vz=3.0 | nbody_system.speed,
        mass=1.0 | nbody_system.mass,
        radius=4.0 | nbody_system.length,
    )
    instance.particles.add_particle(p)

    radii = [x.radius for x in instance.particles]
    expected = [x | nbody_system.length for x in [0.005, 0.005, 4.0]]
    assert all(x == y for x,y in zip(radii, expected)), \
            "cannot add new particle with different radius"



@pytest.mark.parametrize(
    "particle_fixture",
    [particle_inputs_test23],
    indirect=True
)
def test_test23(make_bhtree, particle_fixture):
    instance = make_bhtree(redirection="none")
    particles = particle_fixture
    instance.particles.add_particles(particles)
    instance.commit_particles()

    #instance, particles = bhtree_test23
    instance.evolve_model(0.1 | nbody_system.time)
    assert instance.particles[0].vy <= 0 | nbody_system.speed

    assert_equal_with_reltol(instance.particles[0].x, 0.1 | nbody_system.length, 4)

    instance.particles.new_channel_to(particles).copy()
    particles.vy = 1 | nbody_system.speed
    particles.new_channel_to(instance.particles).copy()

    instance.evolve_model(0.2 | nbody_system.time)

    assert instance.particles[0].vy > 0 | nbody_system.speed
    assert_equal_with_reltol(instance.particles[0].y, 0.1 | nbody_system.length, 4)




