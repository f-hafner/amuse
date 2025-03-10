
# -*- coding: utf-8 -*-
from amusetest import TestWithMPI
import os
import sys
import numpy
import time
import math

from amuse.community.bhtree.interface import BHTreeInterface, BHTree
from amuse.support.exceptions import AmuseException
from amuse.units import constants
from amuse.units import nbody_system
from amuse.units import units
from amuse.units import quantities
from amuse import datamodel
from amuse.datamodel import particle_attributes
from amuse.ic import plummer
try:
    from matplotlib import pyplot
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


from pytest import fixture
from pytest import approx


from amuse.units.quantities import none
from amuse.units.quantities import to_quantity
from amuse.units.quantities import is_quantity


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
def check_equal_with_abstol(x, y, digits, msg=""):
    """Ported from failUnlessAlmostEqual."""
    check_comparable(x, y)
    assert x == approx(y, abs=10**(-digits)), msg


def check_equal_units(x, y, msg="", in_units=None):
    """Ported from failUnlessEqual."""
    check_comparable(x, y)
    x_num, y_num = convert_to_numeric(x, y, in_units)
    assert x_num == y_num, msg



@fixture
def bhtree_setup():
    #convert_nbody = nbody_system.nbody_to_si(1.0 | units.MSun, 149.5e6 | units.km) # for test1
    convert_nbody = nbody_system.nbody_to_si(5.0 | units.kg, 10.0 | units.m) # for test4
    # TODO: I should somehow compose the fixture to take one of the above and do all the operations
    # or alternatively: create separate fixtures for both tests? since they
    # also differ in the entire setup? -> need to think more carefully about a good design
    # that also other people understand.


    instance = BHTree(convert_nbody)
    instance.parameters.epsilon_squared = 0.001 | units.AU**2
    instance.commit_parameters()

    stars = datamodel.Stars(2)

    sun = stars[0]
    sun.mass = units.MSun(1.0)
    sun.position = [0.0, 0.0, 0.0] | units.m
    sun.velocity = [0.0, 0.0, 0.0] | units.ms
    sun.radius = units.RSun(1.0)

    earth = stars[1]
    earth.mass = units.kg(5.9736e24)
    earth.radius = units.km(6371)
    earth.position = [149.5e6, 0.0, 0.0] | units.km
    earth.velocity = [0.0, 29800, 0.0] | units.ms

    instance.particles.add_particles(stars)
    #instance.commit_particles() # this disables adding particles later on
    yield stars, instance

    instance.cleanup_code()
    instance.stop()



def test_test1(bhtree_setup):
    stars, bhtree = bhtree_setup
    sun, earth = stars
    postion_at_start = earth.position.value_in(units.AU)[0]

    bhtree.evolve_model(365.0 | units.day)
    bhtree.particles.copy_values_of_all_attributes_to(stars)

    postion_after_full_rotation = earth.position.value_in(units.AU)[0]
    check_equal_with_abstol(postion_at_start, postion_after_full_rotation, 3)


    bhtree.evolve_model(365.0 + (365.0 / 2) | units.day)
    bhtree.particles.copy_values_of_all_attributes_to(stars)
    postion_after_half_a_rotation = earth.position.value_in(units.AU)[0]

    check_equal_with_abstol(-postion_at_start, postion_after_half_a_rotation, 2)

    bhtree.evolve_model(365.0 + (365.0 / 2) + (365.0 / 4) | units.day)
    bhtree.particles.copy_values_of_all_attributes_to(stars)
    postion_after_half_a_rotation = earth.position.value_in(units.AU)[1]

    check_equal_with_abstol(-postion_at_start, postion_after_half_a_rotation, 1)


def test_test4(bhtree_setup):
    _, bhtree = bhtree_setup

    index = bhtree.new_particle(
        15.0 | units.kg,
        10.0 | units.m, 20.0 | units.m, 30.0 | units.m,
        0.0 | units.m/units.s, 0.0 | units.m/units.s, 0.0 | units.m/units.s,
        10.0 | units.m
    )
    bhtree.commit_particles()
    check_equal_units(bhtree.get_mass(index), 15.0 | units.kg, "new particle not added correctly")
    check_equal_units(bhtree.get_radius(index), 10.0 | units.m)




# NOTES
# test2 seems mostly for the plotting? there's a test for the sun radius,
# but this just tests that attributes of the same instance (sun) is the same?
# test3 also does not test anything
# fct new_system_of_sun_and_earth not used anywhere


class TestBHTree(TestWithMPI):


    def test4(self):
        convert_nbody = nbody_system.nbody_to_si(5.0 | units.kg, 10.0 | units.m)

        instance = BHTree(convert_nbody)
        instance.commit_parameters()
        breakpoint()

        index = instance.new_particle(
            15.0 | units.kg,
            10.0 | units.m, 20.0 | units.m, 30.0 | units.m,
            # 1.0 | units.m/units.s, 1.0 | units.m/units.s, 3.0 | units.m/units.s
            0.0 | units.m/units.s, 0.0 | units.m/units.s, 0.0 | units.m/units.s,
            10.0 | units.m
        )
        instance.commit_particles()
        self.assertEqual(instance.get_mass(index), 15.0 | units.kg)
        self.assertEqual(instance.get_radius(index), 10.0 | units.m)
        instance.cleanup_code()
        instance.stop()

    def test5(self):

        instance = BHTree()
        instance.commit_parameters()

        index = instance.new_particle(
            15.0 | nbody_system.mass,
            10.0 | nbody_system.length, 20.0 | nbody_system.length, 30.0 | nbody_system.length,
            1.0 | nbody_system.speed, 1.0 | nbody_system.speed, 3.0 | nbody_system.speed,
            10.0 | nbody_system.length
        )
        instance.commit_particles()
        self.assertEqual(instance.get_mass(index), 15.0 | nbody_system.mass)
        self.assertEqual(instance.get_radius(index), 10.0 | nbody_system.length)

        instance.cleanup_code()
        instance.stop()

    def test6(self):
        convert_nbody = nbody_system.nbody_to_si(5.0 | units.kg, 10.0 | units.m)
        instance = BHTree(convert_nbody)
        instance.commit_parameters()

        indices = instance.new_particle(
            [15.0, 30.0] | units.kg,
            [10.0, 20.0] | units.m, [20.0, 40.0] | units.m, [30.0, 50.0] | units.m,
            # 1.0 | units.m/units.s, 1.0 | units.m/units.s, 3.0 | units.m/units.s
            [0.0, 0.01] | units.m/units.s, [0.0, 0.01] | units.m/units.s, [0.0, 0.01] | units.m/units.s,
            [10.0, 20.0] | units.m
        )
        instance.commit_particles()

        self.assertEqual(instance.get_mass(indices[0]), 15.0 | units.kg)
        self.assertEqual(instance.get_mass(indices)[0], 15.0 | units.kg)

        self.assertRaises(AmuseException, instance.get_mass, [4, 5],
            expected_message="Error when calling 'get_mass' of a 'BHTree', errorcode is -1")

        instance.cleanup_code()
        instance.stop()

    def test7(self):
        convert_nbody = nbody_system.nbody_to_si(5.0 | units.kg, 10.0 | units.m)

        instance = BHTree(convert_nbody)
        instance.commit_parameters()

        particles = datamodel.Particles(2)
        self.assertEqual(len(instance.particles), 0)

        particles.mass = [15.0, 30.0] | units.kg
        particles.radius = [10.0, 20.0] | units.m
        particles.position = [[10.0, 20.0, 30.0], [20.0, 40.0, 60.0]] | units.m
        particles.velocity = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] | units.m / units.s

        instance.particles.add_particles(particles)
        instance.commit_particles()

        self.assertEqual(instance.get_mass(1), 15.0 | units.kg)
        self.assertAlmostRelativeEquals(instance.get_position(1)[2], 30.0 | units.m)

        self.assertEqual(len(instance.particles), 2)

        self.assertAlmostRelativeEquals(instance.particles.mass[1], 30.0 | units.kg)
        self.assertAlmostRelativeEquals(instance.particles.position[1][2], 60.0 | units.m)
        instance.cleanup_code()
        instance.stop()

    def test8(self):
        convert_nbody = nbody_system.nbody_to_si(5.0 | units.kg, 10.0 | units.m)

        instance = BHTree(convert_nbody)
        instance.commit_parameters()

        particles = datamodel.Particles(2)
        self.assertEqual(len(instance.particles), 0)

        particles.mass = [15.0, 30.0] | units.kg
        particles.radius = [10.0, 20.0] | units.m
        particles.position = [[10.0, 20.0, 30.0], [20.0, 40.0, 60.0]] | units.m
        particles.velocity = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] | units.m / units.s

        instance.particles.add_particles(particles)
        instance.commit_particles()

        instance.particles.mass = [17.0, 33.0] | units.kg

        self.assertEqual(instance.get_mass(1), 17.0 | units.kg)
        instance.cleanup_code()
        instance.stop()

    def test9(self):
        instance = BHTree()
        instance.initialize_code()
        instance.parameters.epsilon_squared = 0.00001 | nbody_system.length**2

        particles = datamodel.Particles(2)
        particles.mass = [1.0, 1.0] | nbody_system.mass
        particles.radius = [0.0001, 0.0001] | nbody_system.length
        particles.position = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]] | nbody_system.length
        particles.velocity = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] | nbody_system.speed
        instance.particles.add_particles(particles)

        zero = 0.0 | nbody_system.length
        fx, fy, fz = instance.get_gravity_at_point(zero, 1.0 | nbody_system.length, zero, zero)
        self.assertAlmostEqual(fx, 0.0 | nbody_system.acceleration, 3)
        self.assertAlmostEqual(fy, 0.0 | nbody_system.acceleration, 3)
        self.assertAlmostEqual(fz, 0.0 | nbody_system.acceleration, 3)

        for x in (0.25, 0.5, 0.75):
            x0 = x | nbody_system.length
            x1 = (2.0 - x) | nbody_system.length
            potential0 = instance.get_potential_at_point(zero, x0, zero, zero)
            potential1 = instance.get_potential_at_point(zero, x1, zero, zero)
            fx0, fy0, fz0 = instance.get_gravity_at_point(zero, x0, zero, zero)
            fx1, fy1, fz1 = instance.get_gravity_at_point(zero, x1, zero, zero)

            self.assertAlmostEqual(fy0, 0.0 | nbody_system.acceleration, 3)
            self.assertAlmostEqual(fz0, 0.0 | nbody_system.acceleration, 3)
            self.assertAlmostEqual(fy1, 0.0 | nbody_system.acceleration, 3)
            self.assertAlmostEqual(fz1, 0.0 | nbody_system.acceleration, 3)

            self.assertAlmostEqual(fx0, -1.0 * fx1, 5)
            fx = (-1.0 / (x0**2) + 1.0 / (x1**2)) * (1.0 | nbody_system.length ** 3 / nbody_system.time ** 2)
            self.assertAlmostEqual(fx, fx0, 2)
            self.assertAlmostEqual(potential0, potential1, 5)
        instance.cleanup_code()
        instance.stop()

    def test10(self):
        instance = BHTree()
        instance.initialize_code()
        instance.parameters.epsilon_squared = 0.00001 | nbody_system.length**2
        instance.commit_parameters()

        particles = datamodel.Particles(6)
        particles.mass = 1.0 | nbody_system.mass
        particles.radius = 0.00001 | nbody_system.length
        particles.position = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]] | nbody_system.length
        particles.velocity = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] | nbody_system.speed
        instance.particles.add_particles(particles)
        instance.commit_particles()

        zero = 0.0 | nbody_system.length
        fx, fy, fz = instance.get_gravity_at_point(zero, zero, zero, zero)
        self.assertAlmostEqual(fx, 0.0 | nbody_system.acceleration, 3)
        self.assertAlmostEqual(fy, 0.0 | nbody_system.acceleration, 3)
        self.assertAlmostEqual(fz, 0.0 | nbody_system.acceleration, 3)

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
                        self.assertAlmostEqual(f0[j], 0.0 | nbody_system.acceleration, 3)
                        self.assertAlmostEqual(f1[j], 0.0 | nbody_system.acceleration, 3)
                    else:
                        self.assertAlmostEqual(f0[j], -1.0 * f1[j], 5)

        instance.stop()

    def test11(self):

        convert_nbody = nbody_system.nbody_to_si(5.0 | units.kg, 10.0 | units.m)

        instance = BHTree(convert_nbody)

        particles = datamodel.Particles(2)
        self.assertEqual(len(instance.particles), 0)

        particles.mass = [15.0, 30.0] | units.kg
        particles.radius = [10.0, 20.0] | units.m
        particles.position = [[10.0, 20.0, 30.0], [20.0, 40.0, 60.0]] | units.m
        particles.velocity = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] | units.m / units.s

        instance.particles.add_particles(particles)

        copyof = instance.particles.copy()

        self.assertAlmostEqual(30 | units.kg, copyof[1].mass, 6)

        copyof[1].mass = 35 | units.kg

        copyof.copy_values_of_all_attributes_to(instance.particles)

        self.assertAlmostEqual(35 | units.kg, instance.particles[1].mass, 6)
        instance.stop()

    def test12(self):

        convert_nbody = nbody_system.nbody_to_si(5.0 | units.kg, 10.0 | units.m)

        instance = BHTree(convert_nbody)
        instance.commit_parameters()

        particles = datamodel.Particles(2)
        self.assertEqual(len(instance.particles), 0)

        particles.mass = [15.0, 30.0] | units.kg
        particles.radius = [10.0, 20.0] | units.m
        particles.position = [[10.0, 20.0, 30.0], [20.0, 40.0, 60.0]] | units.m
        particles.velocity = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] | units.m / units.s

        instance.particles.add_particles(particles)
        instance.commit_particles()

        copyof = instance.particles.copy()

        instance.set_state(1, 16 | units.kg, 20.0 | units.m, 40.0 | units.m, 60.0 | units.m,
                                 1.0 | units.ms, 1.0 | units.ms, 1.0 | units.ms)

        curr_state = instance.get_state(1)
        for expected, actual in zip((16 | units.kg, 20.0 | units.m, 40.0 | units.m, 60.0 | units.m,
                                 1.0 | units.ms, 1.0 | units.ms, 1.0 | units.ms, 0 | units.m), curr_state):
            self.assertAlmostRelativeEquals(actual, expected)

        instance.set_state(1, 16 | units.kg, 20.0 | units.m, 40.0 | units.m, 60.0 | units.m,
                                 1.0 | units.ms, 1.0 | units.ms, 1.0 | units.ms, 20.0 | units.m)

        curr_state = instance.get_state(1)
        for expected, actual in zip((16 | units.kg, 20.0 | units.m, 40.0 | units.m, 60.0 | units.m,
                                 1.0 | units.ms, 1.0 | units.ms, 1.0 | units.ms, 20 | units.m), curr_state):
            self.assertAlmostRelativeEquals(actual, expected)

        instance.stop()

    def test13(self):

        convert_nbody = nbody_system.nbody_to_si(1.0 | units.kg, 1.0 | units.m)

        instance = BHTree(convert_nbody)
        instance.commit_parameters()

        particles = datamodel.Particles(2)
        self.assertEqual(len(instance.particles), 0)

        particles.mass = [30.0, 30.0] | units.kg
        particles.radius = [1.0, 1.0] | units.m
        particles.position = [[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]] | units.m
        particles.velocity = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] | units.m / units.s

        instance.particles.add_particles(particles)
        instance.commit_particles()

        copyof = instance.particles.copy()

        com = instance.center_of_mass_position
        self.assertAlmostEqual(com[0], quantities.new_quantity(0.0, units.m), constants.precision)
        instance.stop()

    def test14(self):
        print("Test14: Testing BHTree parameters (I)")
        convert_nbody = nbody_system.nbody_to_si(1.0 | units.yr, 1.0 | units.AU)
        instance = BHTree(convert_nbody)

        value, error = instance.legacy_interface.get_epsilon_squared()
        self.assertEqual(0, error)
        self.assertEqual(0.125, value)
        self.assertAlmostEqual(0.125 | units.AU**2, instance.parameters.epsilon_squared, in_units=units.AU**2)
        for x in [0.01, 0.1, 0.2]:
            instance.parameters.epsilon_squared = x | units.AU**2
            self.assertAlmostEqual(x | units.AU**2, instance.parameters.epsilon_squared, in_units=units.AU**2)

        (value, error) = instance.legacy_interface.get_time_step()
        self.assertEqual(0, error)
        self.assertEqual(0.015625, value)
        self.assertAlmostEqual(0.015625 | units.yr, instance.parameters.timestep, in_units=units.yr)
        for x in [0.001, 0.01, 0.1]:
            instance.parameters.timestep = x | units.yr
            self.assertAlmostEqual(x | units.yr, instance.parameters.timestep, in_units=units.yr)

        (value, error) = instance.legacy_interface.get_theta_for_tree()
        self.assertEqual(0, error)
        self.assertEqual(0.75, value)
        self.assertEqual(0.75, instance.parameters.opening_angle)
        for x in [0.2, 0.5, 0.7]:
            instance.parameters.opening_angle = x
            self.assertEqual(x, instance.parameters.opening_angle)

        (value, error) = instance.legacy_interface.get_use_self_gravity()
        self.assertEqual(0, error)
        self.assertEqual(1, value)
        self.assertEqual(1, instance.parameters.use_self_gravity)
        for x in [0, 1]:
            instance.parameters.use_self_gravity = x
            self.assertEqual(x, instance.parameters.use_self_gravity)

        (value, error) = instance.legacy_interface.get_ncrit_for_tree()
        self.assertEqual(0, error)
        self.assertEqual(12, value)
        self.assertEqual(12, instance.parameters.ncrit_for_tree)
        for x in [512, 2048, 4096]:
            instance.parameters.ncrit_for_tree = x
            self.assertEqual(x, instance.parameters.ncrit_for_tree)

        (value, error) = instance.legacy_interface.get_dt_dia()
        self.assertEqual(0, error)
        self.assertEqual(1.0, value)
        self.assertAlmostEqual(1.0 | units.yr, instance.parameters.dt_dia, in_units=units.yr)
        for x in [0.1, 10.0, 100.0]:
            instance.parameters.dt_dia = x | units.yr
            self.assertAlmostEqual(x | units.yr, instance.parameters.dt_dia, in_units=units.yr)
        instance.stop()

    def test15(self):
        print("Test15: Testing effect of BHTree parameter epsilon_squared")
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
        for log_eps2 in range(-9, 10, 2):
            instance = BHTree(convert_nbody)
            instance.initialize_code()
            instance.parameters.epsilon_squared = 10.0**log_eps2 | units.AU ** 2
            instance.particles.add_particles(particles)
            instance.commit_particles()
            instance.evolve_model(0.25 | units.yr)
            final_direction.append(math.atan((instance.particles[1].velocity[0] /
                instance.particles[1].velocity[1])))
            instance.stop()
        # Small values of epsilon_squared should result in normal earth-sun dynamics: rotation of 90 degrees
        self.assertAlmostEqual(abs(final_direction[0]), abs(initial_direction+math.pi/2.0), 2)
        # Large values of epsilon_squared should result in ~ no interaction
        self.assertAlmostEqual(final_direction[-1], initial_direction, 2)
        # Outcome is most sensitive to epsilon_squared when epsilon_squared = d(earth, sun)^2
        delta = [abs(final_direction[i+1]-final_direction[i]) for i in range(len(final_direction)-1)]
        self.assertEqual(delta[len(final_direction)//2 - 1], max(delta))

    def test16(self):
        numpy.random.seed(0)
        number_of_stars = 2
        stars = plummer.new_plummer_model(number_of_stars)
        stars.radius = 0.00001 | nbody_system.length
        stars.scale_to_standard()

        instance = BHTree()
        instance.initialize_code()
        instance.parameters.epsilon_squared = (1.0 / 20.0 / (number_of_stars**0.33333) | nbody_system.length)**2
        instance.parameters.timestep = 0.004 | nbody_system.time
        instance.parameters.timestep = 0.00001 | nbody_system.time
        instance.commit_parameters()
        print(instance.parameters.timestep)
        instance.particles.add_particles(stars)
        instance.commit_particles()
        energy_total_t0 = instance.potential_energy + instance.kinetic_energy
        request = instance.evolve_model.asynchronous(1.0 | nbody_system.time)
        request.result()
        energy_total_t1 = instance.potential_energy + instance.kinetic_energy

        self.assertAlmostRelativeEqual(energy_total_t0, energy_total_t1, 3)
        instance.stop()
        numpy.random.seed()

    def test17(self):
        print("Testing BHTree collision_detection")
        particles = datamodel.Particles(7)
        particles.mass = 0.001 | nbody_system.mass
        particles.radius = 0.01 | nbody_system.length
        particles.x = [-101.0, -100.0, -0.5, 0.5, 100.0, 101.0, 104.0] | nbody_system.length
        particles.y = 0 | nbody_system.length
        particles.z = 0 | nbody_system.length
        particles.velocity = [[2, 0, 0], [-2, 0, 0]]*3 + [[-4, 0, 0]] | nbody_system.speed

        instance = BHTree(redirection='none')
        instance.initialize_code()
        instance.parameters.set_defaults()

        # Uncommenting any of the following two lines will suppress collision detection
# ~        instance.parameters.use_self_gravity = 0
# ~        instance.parameters.epsilon_squared = 0.0 | nbody_system.length**2

        instance.parameters.opening_angle = 0.1
        instance.particles.add_particles(particles)
        collisions = instance.stopping_conditions.collision_detection
        collisions.enable()
        instance.evolve_model(1.0 | nbody_system.time)

        self.assertTrue(collisions.is_set())
        self.assertTrue(instance.model_time < 0.5 | nbody_system.time)
        self.assertEqual(len(collisions.particles(0)), 3)
        self.assertEqual(len(collisions.particles(1)), 3)
        self.assertEqual(len(particles - collisions.particles(0) - collisions.particles(1)), 1)
        self.assertEqual(abs(collisions.particles(0).x - collisions.particles(1).x) <
                (collisions.particles(0).radius + collisions.particles(1).radius),
                [True, True, True])

        sticky_merged = datamodel.Particles(len(collisions.particles(0)))
        sticky_merged.mass = collisions.particles(0).mass + collisions.particles(1).mass
        sticky_merged.radius = collisions.particles(0).radius
        for p1, p2, merged in zip(collisions.particles(0), collisions.particles(1), sticky_merged):
            merged.position = (p1 + p2).center_of_mass()
            merged.velocity = (p1 + p2).center_of_mass_velocity()

        print(instance.model_time)
        print(instance.particles)
        instance.particles.remove_particles(collisions.particles(0) + collisions.particles(1))
        instance.particles.add_particles(sticky_merged)

        instance.evolve_model(1.0 | nbody_system.time)
        print()
        print(instance.model_time)
        print(instance.particles)
        self.assertTrue(collisions.is_set())
        self.assertTrue(instance.model_time < 1.0 | nbody_system.time)
        self.assertEqual(len(collisions.particles(0)), 1)
        self.assertEqual(len(collisions.particles(1)), 1)
        self.assertEqual(len(instance.particles - collisions.particles(0) - collisions.particles(1)), 2)
        self.assertEqual(abs(collisions.particles(0).x - collisions.particles(1).x) <
                (collisions.particles(0).radius + collisions.particles(1).radius),
                [True])
        instance.stop()

    def test18(self):
        particles = datamodel.Particles(2)
        particles.x = [0.0, 10.0] | nbody_system.length
        particles.y = 0 | nbody_system.length
        particles.z = 0 | nbody_system.length
        particles.radius = 0.005 | nbody_system.length
        particles.vx = 0 | nbody_system.speed
        particles.vy = 0 | nbody_system.speed
        particles.vz = 0 | nbody_system.speed
        particles.mass = 1.0 | nbody_system.mass

        instance = BHTree()
        instance.initialize_code()
        instance.parameters.stopping_conditions_number_of_steps = 2
        self.assertEqual(instance.parameters.stopping_conditions_number_of_steps, 2)
        instance.parameters.epsilon_squared = (0.01 | nbody_system.length)**2
        instance.particles.add_particles(particles)
        instance.stopping_conditions.number_of_steps_detection.enable()
        instance.evolve_model(10 | nbody_system.time)
        self.assertTrue(instance.stopping_conditions.number_of_steps_detection.is_set())
        self.assertTrue(instance.model_time < 10 | nbody_system.time)
        instance.stop()

    def test19(self):
        particles = datamodel.Particles(2)
        particles.x = [0.0, 10.0] | nbody_system.length
        particles.y = 0.0 | nbody_system.length
        particles.z = 0.0 | nbody_system.length
        particles.radius = 0.005 | nbody_system.length
        particles.vx = 0.0 | nbody_system.speed
        particles.vy = 0.0 | nbody_system.speed
        particles.vz = 0.0 | nbody_system.speed
        particles.mass = 1.0 | nbody_system.mass

        very_short_time_to_evolve = 1 | units.s
        very_long_time_to_evolve = 1e9 | nbody_system.time

        instance = BHTree()
        instance.initialize_code()
        instance.parameters.stopping_conditions_timeout = very_short_time_to_evolve
        self.assertEqual(instance.parameters.stopping_conditions_timeout, very_short_time_to_evolve)
        instance.parameters.epsilon_squared = (0.01 | nbody_system.length)**2
        instance.particles.add_particles(particles)
        instance.stopping_conditions.timeout_detection.enable()
        start = time.time()
        instance.evolve_model(very_long_time_to_evolve)
        end = time.time()
        self.assertTrue(instance.stopping_conditions.timeout_detection.is_set())
        self.assertTrue((end-start) < very_short_time_to_evolve.value_in(units.s) + 2)  # 2 = some overhead compensation
        instance.stop()

    def test20(self):
        particles = datamodel.Particles(2)
        particles.x = [0.0, 10.0] | nbody_system.length
        particles.y = 0.0 | nbody_system.length
        particles.z = 0.0 | nbody_system.length
        particles.radius = 0.005 | nbody_system.length
        particles.vx = 0.0 | nbody_system.speed
        particles.vy = 0.0 | nbody_system.speed
        particles.vz = 0.0 | nbody_system.speed
        particles.mass = 1.0 | nbody_system.mass

        very_short_time_to_evolve = 1 | units.s
        very_long_time_to_evolve = 1e9 | nbody_system.time

        instance = BHTree()
        instance.initialize_code()
        instance.parameters.stopping_conditions_timeout = very_short_time_to_evolve
        self.assertEqual(instance.parameters.stopping_conditions_timeout, very_short_time_to_evolve)
        instance.parameters.epsilon_squared = (0.01 | nbody_system.length)**2
        instance.particles.add_particles(particles)
        codeparticles1 = instance.particles
        instance.particles.add_particle(datamodel.Particle(
            position=[0, 1, 2] | nbody_system.length,
            velocity=[0, 0, 0] | nbody_system.speed,
            radius=0.005 | nbody_system.length,
            mass=1 | nbody_system.mass
        ))
        codeparticles2 = instance.particles
        self.assertTrue(codeparticles1 is codeparticles2)
        instance.cleanup_code()
        codeparticles3 = instance.particles
        self.assertFalse(codeparticles1 is codeparticles3)

        instance.stop()

    def test21(self):
        particles = datamodel.Particles(2)
        particles.x = [0.0, 10.0] | nbody_system.length
        particles.y = 0.0 | nbody_system.length
        particles.z = 0.0 | nbody_system.length
        particles.radius = 0.005 | nbody_system.length
        particles.vx = 0.0 | nbody_system.speed
        particles.vy = 0.0 | nbody_system.speed
        particles.vz = 0.0 | nbody_system.speed
        particles.mass = 1.0 | nbody_system.mass

        very_short_time_to_evolve = 1 | units.s
        very_long_time_to_evolve = 1e9 | nbody_system.time

        instance = BHTree()
        instance.initialize_code()
        instance.parameters.epsilon_squared = (1e-5 | nbody_system.length)**2
        instance.particles.add_particles(particles)
        instance.commit_particles()
        self.assertAlmostRelativeEquals(instance.potential_energy, -0.1 | nbody_system.energy, 5)
        instance.stop()

    def test22(self):
        particles = datamodel.Particles(2)
        particles.x = [0.0, 10.0] | nbody_system.length
        particles.y = 0.0 | nbody_system.length
        particles.z = 0.0 | nbody_system.length
        particles.vx = 0.0 | nbody_system.speed
        particles.vy = 0.0 | nbody_system.speed
        particles.vz = 0.0 | nbody_system.speed
        particles.mass = 1.0 | nbody_system.mass

        instance = BHTree()
        instance.particles.add_particles(particles)
        instance.commit_particles()
        self.assertEqual(instance.particles[0].radius, 0.0 | nbody_system.length)
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
        self.assertEqual(instance.particles[0].radius, 0.0 | nbody_system.length)
        self.assertEqual(instance.particles[1].radius, 0.0 | nbody_system.length)
        self.assertEqual(instance.particles[2].radius, 4.0 | nbody_system.length)

        instance.stop()

    def test23(self):
        particles = datamodel.Particles(2)
        particles.x = [0.0, 10.0] | nbody_system.length
        particles.y = 0.0 | nbody_system.length
        particles.z = 0.0 | nbody_system.length
        particles.vx = 1.0 | nbody_system.speed
        particles.vy = 0.0 | nbody_system.speed
        particles.vz = 0.0 | nbody_system.speed
        particles.mass = 0.1 | nbody_system.mass

        instance = BHTree(redirection="none")
        instance.particles.add_particles(particles)
        instance.commit_particles()
        instance.evolve_model(0.1 | nbody_system.time)
        self.assertFalse(instance.particles[0].vy > 0 | nbody_system.speed)
        self.assertAlmostRelativeEquals(instance.particles[0].x, 0.1 | nbody_system.length, 4)
        instance.particles.new_channel_to(particles).copy()
        particles.vy = 1 | nbody_system.speed
        particles.new_channel_to(instance.particles).copy()

        instance.evolve_model(0.2 | nbody_system.time)
        self.assertTrue(instance.particles[0].vy > 0 | nbody_system.speed)
        self.assertAlmostRelativeEquals(instance.particles[0].y, 0.1 | nbody_system.length, 4)
        instance.stop()
