import pytest
import math
import logging
import numpy as np

from amuse.support.exceptions import AmuseException, CoreException
from amuse.units import nbody_system
from amuse.units import units
from amuse.units import quantities

from amuse import datamodel
from amuse.ic import plummer
from amuse.support.testing.equality_with_units import assert_equal
from amuse.support.testing.equality_with_units import assert_equal_with_abstol
from amuse.support.testing.equality_with_units import assert_equal_with_reltol

from .fixtures import *
from . import initial_conditions as ic

logger = logging.getLogger(__name__)

def _set_timestep_parameters(instance, timestep_param: tuple):
    """Helper to set timestep params for nbody code.

    Args:
        instance: N-body code instance
        timestep_param: Tuple containing (parameter_name, parameter_value)

    Returns:
        The instance with updated timestep parameters

    Raises:
        AttributeError: If parameter can't be set
    """
    try:
        name, value = timestep_param
        setattr(instance.parameters, name, value)
    except AttributeError as err:
        msg = """No valid parameters for timesteps provided.
                Have you correctly defined an `nbody_timestep_parameter`
                in your package's `conftest.py`?"""
        raise AttributeError(msg) from err
    return instance



# Note how nbody_instance and nbody_instance_kg fixtures are passed as strings
    # and evaluated with request.getfixturevalue. See also:
    # https://miguendes.me/how-to-use-fixtures-as-arguments-in-pytestmarkparametrize
    # https://github.com/pytest-dev/pytest/issues/349
@pytest.mark.parametrize(
        ("nbody_input", "attribute_list_particle"),
        [("nbody_instance", ic.list_new_particle),
         ("nbody_instance_kg", ic.list_new_particle_kg)]
        )
def test_new_particle(nbody_input, attribute_list_particle, request):
    """Test adding a new particle to N-body code and verifying mass and radius."""
    nbody_instance = request.getfixturevalue(nbody_input)
    index = nbody_instance.new_particle(*attribute_list_particle)
    nbody_instance.commit_particles()
    assert_equal(nbody_instance.get_mass(index), attribute_list_particle[0])
    assert_equal(nbody_instance.get_radius(index), attribute_list_particle[1])


def test_multiple_new_particles_index(nbody_instance_kg):
    """Test adding multiple particles at once and retrieval by index."""
    instance = nbody_instance_kg
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

    expected_typename = type(instance).__name__
    expected_string = f"Error when calling 'get_mass' of a '{expected_typename}', errorcode is -1"
    assert expected_string in str(excinfo.value)


@pytest.mark.parametrize(
    "particle_fixture, raw_particle_data",
    [(ic.kg_particle, ic.kg_particle)],
    indirect=["particle_fixture"]
)
def test_multiple_new_particles(nbody_instance_kg, particle_fixture, raw_particle_data, starting_particle_index):
    """Test adding multiple particles verifying all attributes match expected values."""
    instance = nbody_instance_kg
    instance.particles.add_particles(particle_fixture)
    instance.commit_particles()

    expected_count = raw_particle_data[0]
    assert len(instance.particles) == expected_count

    expected_mass = raw_particle_data[1]["mass"]
    for idx in range(expected_count):
        assert_equal(instance.get_mass(starting_particle_index + idx), expected_mass[idx])

    expected_radius = raw_particle_data[1]["radius"]
    for idx in range(expected_count):
        assert_equal(instance.get_radius(starting_particle_index + idx), expected_radius[idx])

    expected_position = raw_particle_data[1]["position"]
    for idx in range(expected_count):
        pos = instance.get_position(starting_particle_index + idx)
        pos_e = expected_position[idx]
        for x, x_e in zip(pos, pos_e):
            assert_equal_with_reltol(x, x_e)

    expected_velocity = raw_particle_data[1]["velocity"]
    for idx in range(expected_count):
        vel = instance.get_velocity(starting_particle_index + idx)
        vel_e = expected_velocity[idx]
        for x, x_e in zip(vel, vel_e):
            assert_equal_with_reltol(x, x_e)


@pytest.mark.parametrize(
    "particle_fixture",
    [ic.kg_particle],
    indirect=True
)
def test_change_existing_particles(nbody_instance_kg, particle_fixture, starting_particle_index):
    """Test modifying attributes of existing particles."""
    instance = nbody_instance_kg

    instance.particles.add_particles(particle_fixture)
    instance.commit_particles()

    instance.particles.mass = [17.0, 33.0] | units.kg
    assert_equal(instance.get_mass(starting_particle_index), 17.0 | units.kg)


# NOTE: 0.125 is the default value for epsilon_squared;
# it is put here to make things explicit.
@pytest.mark.parametrize(
    ("particle_fixture", "point", "epsilon2"),
    [(ic.gravity_with_same_potential, 1.0, 0.00001),
     (ic.gravity_at_positions, 0.0, 0.125)],
    indirect=["particle_fixture"]
)
def test_zero_gravity(nbody_instance, particle_fixture, point, epsilon2):
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
    [ic.gravity_with_same_potential],
    indirect=True
)
@pytest.mark.parametrize("x", [0.25, 0.5, 0.75])
def test_gravity_with_same_potential(nbody_instance, particle_fixture, x):
    """Test equal potential at symmetric positions and expected gravity values."""
    instance = nbody_instance
    instance.parameters.epsilon_squared = 0.00001 | nbody_system.length**2
    instance.particles.add_particles(particle_fixture)

    zero = 0.0 | nbody_system.length

    x0 = x | nbody_system.length
    x1 = (2.0 - x) | nbody_system.length
    potential0 = instance.get_potential_at_point(zero, x0, zero, zero)
    potential1 = instance.get_potential_at_point(zero, x1, zero, zero)
    assert_equal_with_reltol(potential0, potential1, 5)

    gravity0 = instance.get_gravity_at_point(zero, x0, zero, zero)
    gravity1 = instance.get_gravity_at_point(zero, x1, zero, zero)
    for i in range(len(gravity0)):
        if i == 0:
            fx0_expected = (-1.0 / (x0**2) + 1.0 / (x1**2)) * (1.0 | nbody_system.length ** 3 / nbody_system.time ** 2)
            assert_equal_with_reltol(gravity0[i], fx0_expected, 2)
            assert_equal_with_reltol(gravity0[i], -1.0 * gravity1[i], 5)
            pass
        else:
            assert_equal_with_reltol(gravity0[i], 0 | nbody_system.acceleration, 3)
            assert_equal_with_reltol(gravity1[i], 0 | nbody_system.acceleration, 3)


@pytest.mark.parametrize(
    "particle_fixture",
    [ic.gravity_at_positions],
    indirect=True
)
@pytest.mark.parametrize("position", [0.25, 0.5, 0.75])
def test_gravity_at_positions(nbody_instance, particle_fixture, position):
    """Test gravity at various positions along each dimension."""
    instance = nbody_instance
    instance.particles.add_particles(particle_fixture)
    instance.commit_particles()

    zero = 0.0 | nbody_system.length
    p0 = position | nbody_system.length
    p1 = -position | nbody_system.length

    for i in range(3):
        args0 = [zero] * 4
        args1 = [zero] * 4
        args0[1 + i] = p0
        args1[1 + i] = p1
        gravity0 = instance.get_gravity_at_point(*args0)
        gravity1 = instance.get_gravity_at_point(*args1)

        for j in range(3):
            if j != i:
                assert_equal_with_reltol(gravity0[j], 0.0 | nbody_system.acceleration, 3)
                assert_equal_with_reltol(gravity1[j], 0.0 | nbody_system.acceleration, 3)
            else:
                assert_equal_with_reltol(gravity0[j], -1.0 * gravity1[j], 5)


@pytest.mark.parametrize(
    "particle_fixture, raw_particle_data",
    [(ic.kg_particle, ic.kg_particle)],
    indirect=["particle_fixture"]
    )
def test_copy_particle_mass(nbody_instance_kg, particle_fixture, raw_particle_data):
    """Test particle copying and attribute copying through channels."""
    instance = nbody_instance_kg
    instance.particles.add_particles(particle_fixture)

    copyof = instance.particles.copy()
    # Note that indexing with instance.particles[idx].mass is all 0-based
    # whereas instance.get_mass(idx) can be 0 or 1-based (see other tests)
    id_to_check = 1
    expected_mass = raw_particle_data[1]["mass"][id_to_check]
    assert_equal_with_reltol(copyof[id_to_check].mass, expected_mass, 6)

    copyof[id_to_check].mass = 35 | units.kg
    channel = copyof.new_channel_to(instance.particles)
    channel.copy_attributes(["mass"])
    assert_equal_with_reltol(instance.particles[id_to_check].mass, 35 | units.kg, 6)


@pytest.mark.parametrize(
    "particle_fixture",
    [ic.kg_particle],
    indirect=True
)
def test_set_state(nbody_instance_kg, particle_fixture, starting_particle_index):
    """Test setting and getting particle state values."""
    instance = nbody_instance_kg

    instance.particles.add_particles(particle_fixture)
    instance.commit_particles()

    state_values = [16 | units.kg,
             20.0 | units.m,
             40.0 | units.m,
             60.0 | units.m,
             1.0 | units.ms,
             1.0 | units.ms,
             1.0 | units.ms]
    instance.set_state(starting_particle_index, *state_values)
    expected_values = state_values + [0 | units.m]

    curr_state = instance.get_state(starting_particle_index)
    for expected, actual in zip(expected_values, curr_state):
        assert_equal_with_reltol(actual, expected)


@pytest.mark.parametrize(
    "particle_fixture",
    [ic.center_of_mass_position],
    indirect=True
)
def test_center_of_mass_position(nbody_instance_kg, particle_fixture):
    """Test calculation of center of mass position."""
    instance = nbody_instance_kg

    instance.particles.add_particles(particle_fixture)
    instance.commit_particles()

    com = instance.center_of_mass_position
    expected = quantities.new_quantity(0.0, units.m)
    assert_equal_with_reltol(com[0], expected)

def test_softening(make_nbody_instance, nbody_timestep_parameter):
    """Test effects of different softening parameters on gravitational dynamics."""
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
    earth.velocity = [2.0*np.pi, -0.0001, 0.0] | units.AU / units.yr

    initial_direction = math.atan((earth.velocity[0]/earth.velocity[1]))
    final_direction = []
    for log_eps2 in range(-9, 10, 2):
        instance = make_nbody_instance(convert_nbody)
        instance.initialize_code()
        _set_timestep_parameters(instance, nbody_timestep_parameter)
        instance.parameters.epsilon_squared = 10.0**log_eps2 | units.AU ** 2
        instance.particles.add_particles(particles)
        instance.commit_particles()
        instance.evolve_model(0.25 | units.yr)
        final_direction.append(math.atan((instance.particles[1].velocity[0] /
            instance.particles[1].velocity[1])))

    # Tests
    # Small values of epsilon_squared should result in normal earth-sun dynamics: rotation of 90 degrees
    assert_equal_with_abstol(abs(final_direction[0]), abs(initial_direction + math.pi/2.0), 2)

    # Large values of epsilon_squared should result in ~ no interaction
    assert_equal_with_abstol(final_direction[-1], initial_direction, 2)

    # Outcome is most sensitive to epsilon_squared when epsilon_squared = d(earth, sun)^2
    delta = [abs(final_direction[i+1]-final_direction[i]) for i in range(len(final_direction)-1)]
    assert max(delta) == delta[len(final_direction)//2 - 1]


@pytest.mark.parametrize(
    "particle_fixture",
    [ic.collision_detection],
    indirect=True
)
def test_collision_detection(make_nbody_instance, particle_fixture):
    """Test collision detection and handling with sticky merges."""
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
    n_particles_start = len(instance.particles)

    max_collisions = 3
    n_collisions = 0
    for _ in range(3):
        instance.evolve_model(1.0 | nbody_system.time)

        assert collisions.is_set(), "collisions not set"
        assert instance.model_time < 0.5 | nbody_system.time, "time too big"

        n_current_collisions = len(collisions.particles(0))
        assert len(collisions.particles(1)) == n_current_collisions, "mismatch in N particles"
        assert len(instance.particles - collisions.particles(0) - collisions.particles(1)) == n_particles_start - 2*n_current_collisions - n_collisions

        left = abs(collisions.particles(0).x - collisions.particles(1).x)
        right = collisions.particles(0).radius + collisions.particles(1).radius
        assert all(left < right)

        sticky_merged = datamodel.Particles(len(collisions.particles(0)))
        sticky_merged.mass = collisions.particles(0).mass + collisions.particles(1).mass
        sticky_merged.radius = collisions.particles(0).radius
        for p1, p2, merged in zip(collisions.particles(0), collisions.particles(1), sticky_merged):
            merged.position = (p1 + p2).center_of_mass()
            merged.velocity = (p1 + p2).center_of_mass_velocity()

        instance.particles.remove_particles(collisions.particles(0) + collisions.particles(1))
        instance.particles.add_particles(sticky_merged)

        n_collisions += n_current_collisions
        # break in first iteration if all collisions are detected at once
        # such as in bhtree
        if n_collisions >= max_collisions:
            break

    instance.evolve_model(1.0 | nbody_system.time)

    assert collisions.is_set(), "collisions not set"
    assert instance.model_time < 1.0 | nbody_system.time, "time too big"
    assert len(collisions.particles(0)) == 1, "mismatch in N particles"
    assert len(collisions.particles(1)) == 1, "mismatch in N particles"
    assert len(instance.particles - collisions.particles(0) - collisions.particles(1)) == 2

    left = abs(collisions.particles(0).x - collisions.particles(1).x)
    right = collisions.particles(0).radius + collisions.particles(1).radius
    assert all(left < right)

@pytest.mark.parametrize(
    "particle_fixture",
    [ic.stop_n_steps],
    indirect=True
)
def test_cleanup(nbody_instance, particle_fixture):
    """Test code cleanup and verification of particle set references."""
    instance = nbody_instance
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
    [ic.stop_n_steps],
    indirect=True
)
def test_potential_energy(nbody_instance, particle_fixture):
    """Test calculation of potential energy."""
    instance = nbody_instance
    instance.particles.add_particles(particle_fixture)

    instance.parameters.epsilon_squared = (1e-5 | nbody_system.length)**2
    assert_equal_with_reltol(instance.potential_energy, -0.1 | nbody_system.energy, 5)


@pytest.mark.parametrize(
    "particle_fixture",
    [ic.stop_n_steps],
    indirect=True
)
def test_add_particle_with_new_radius(nbody_instance, particle_fixture):
    """Test adding a particle with a different radius than existing particles."""
    instance = nbody_instance
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
    "particle_fixture, raw_particle_data",
    [(ic.direction_and_speed_when_evolving_model, ic.direction_and_speed_when_evolving_model)],
    indirect=["particle_fixture"]
    )
def test_direction_and_speed_when_evolving_model(make_nbody_instance, particle_fixture, raw_particle_data):
    """Test particle movement direction and speed during model evolution."""
    instance = make_nbody_instance(redirection="none")
    particles = particle_fixture
    instance.particles.add_particles(particles)
    instance.commit_particles()

    idx_to_check = 0
    x_start = raw_particle_data[1]["x"][idx_to_check]
    y_start = raw_particle_data[1]["y"][idx_to_check]

    instance.evolve_model(0.1 | nbody_system.time)
    assert instance.particles[idx_to_check].vy <= 0 | nbody_system.speed
    assert instance.particles[idx_to_check].x > x_start

    instance.particles.new_channel_to(particles).copy()
    particles.vy = 1 | nbody_system.speed
    particles.new_channel_to(instance.particles).copy()

    instance.evolve_model(0.2 | nbody_system.time)

    assert instance.particles[idx_to_check].vy > 0 | nbody_system.speed
    assert instance.particles[idx_to_check].y > y_start


def test_system_sun_earth(make_nbody_instance):
    """Test Sun-Earth orbital evolution over multiple time periods."""
    # Set up the instance
    convert_nbody = nbody_system.nbody_to_si(1.0 | units.MSun, 149.5e6 | units.km)
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
    assert_equal_with_abstol(position_at_start, position_after_full_rotation, 2)

    instance.evolve_model(365.0 + (365.0 / 2) | units.day)
    instance.particles.copy_values_of_all_attributes_to(stars)
    position_after_half_a_rotation = earth.position.value_in(units.AU)[0]


    position_delta = position_at_start + position_after_half_a_rotation
    assert_equal_with_abstol(position_delta, 0.00, 2)

    instance.evolve_model(365.0 + (365.0 / 2) + (365.0 / 4) | units.day)
    instance.particles.copy_values_of_all_attributes_to(stars)
    position_after_half_a_rotation = earth.position.value_in(units.AU)[1]

    assert_equal_with_abstol(-position_at_start, position_after_half_a_rotation, 1)


def test_energy_unchanged(make_nbody_instance, nbody_timestep_parameter):
    """Test energy conservation during model evolution."""
    # Setup
    np.random.seed(0)
    number_of_stars = 2
    stars = plummer.new_plummer_model(number_of_stars)
    stars.radius = 0.00001 | nbody_system.length
    stars.scale_to_standard()

    instance = make_nbody_instance()
    instance.initialize_code()
    instance.parameters.epsilon_squared = (1.0 / 20.0 / (number_of_stars**0.33333) | nbody_system.length)**2
    instance = _set_timestep_parameters(instance, nbody_timestep_parameter)

    instance.commit_parameters()
    instance.particles.add_particles(stars)
    instance.commit_particles()

    # Test
    energy_total_t0 = instance.potential_energy + instance.kinetic_energy
    request = instance.evolve_model.asynchronous(1.0 | nbody_system.time)
    request.result()
    energy_total_t1 = instance.potential_energy + instance.kinetic_energy

    assert_equal_with_reltol(energy_total_t0, energy_total_t1, 3)

@pytest.mark.parametrize("n_workers", [1, 4])
def test_energy_changed(make_nbody_instance, n_workers, nbody_timestep_parameter):
    """Test energy changes when particle masses are modified during evolution."""
    # Setup
    instance = make_nbody_instance(number_of_workers=n_workers)
    instance.initialize_code()

    instance.parameters.epsilon_squared = 0.0 | nbody_system.length**2
    instance = _set_timestep_parameters(instance, nbody_timestep_parameter)

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


def test_states(nbody_instance, make_nbody_instance, nbody_timestep_parameter):
    """Test code state transitions during different operations."""
    stars = plummer.new_plummer_model(100)
    black_hole = datamodel.Particle()
    black_hole.mass = 1.0 | nbody_system.mass
    black_hole.radius = 0.0 | nbody_system.length
    black_hole.position = [0.0, 0.0, 0.0] | nbody_system.length
    black_hole.velocity = [0.0, 0.0, 0.0] | nbody_system.speed

    instance = nbody_instance
    assert instance.get_name_of_current_state() == "UNINITIALIZED"
    instance.initialize_code()
    assert instance.get_name_of_current_state() == "INITIALIZED"
    instance.parameters.epsilon_squared = 0.0 | nbody_system.length**2
    instance = _set_timestep_parameters(instance, nbody_timestep_parameter)

    instance.commit_parameters()
    assert instance.get_name_of_current_state() == "EDIT"
    instance.particles.add_particles(stars)
    instance.commit_particles()
    assert instance.get_name_of_current_state() == "RUN"
    instance.particles.remove_particle(stars[0])
    instance.particles.add_particle(black_hole)
    assert instance.get_name_of_current_state() == "UPDATE"
    instance.recommit_particles()
    assert instance.get_name_of_current_state() == "RUN"
    instance.evolve_model(0.001 | nbody_system.time)
    assert instance.get_name_of_current_state() == "EVOLVED"
    instance.synchronize_model()
    assert instance.get_name_of_current_state() == "RUN"

    instance = make_nbody_instance()
    assert instance.get_name_of_current_state() == "UNINITIALIZED"
    instance.parameters.epsilon_squared = 0.0 | nbody_system.length**2
    instance = _set_timestep_parameters(instance, nbody_timestep_parameter)

    assert instance.get_name_of_current_state() == "INITIALIZED"
    instance.particles.add_particles(stars)
    assert instance.get_name_of_current_state() == "EDIT"
    _ = instance.particles[0].mass
    assert instance.get_name_of_current_state() == "RUN"
    instance.particles.remove_particle(stars[0])
    instance.particles.add_particle(black_hole)
    assert instance.get_name_of_current_state() == "UPDATE"
    _ = instance.particles[0].mass
    assert instance.get_name_of_current_state() == "RUN"
    instance.evolve_model(0.001 | nbody_system.time)
    assert instance.get_name_of_current_state() == "EVOLVED"
    _ = instance.particles[0].mass
    assert instance.get_name_of_current_state() == "RUN"

def test_potential_with_multiple_workers(make_nbody_instance):
    """Test consistent potential calculation with different worker counts."""
    particles = plummer.new_plummer_model(200)
    particles.scale_to_standard()
    instance = make_nbody_instance()
    instance.initialize_code()
    instance.parameters.epsilon_squared = 0.00000 | nbody_system.length**2
    instance.particles.add_particles(particles)

    x = np.arange(-1, 1, 0.1) | nbody_system.length
    zero = np.zeros(len(x)) | nbody_system.length
    potential0 = instance.get_potential_at_point(zero, x, zero, zero)
    for n in (2, 3, 4):
        temp_instance = make_nbody_instance(number_of_workers=n)
        temp_instance.initialize_code()
        temp_instance.parameters.epsilon_squared = 0.00000 | nbody_system.length**2
        temp_instance.particles.add_particles(particles)
        potential = temp_instance.get_potential_at_point(zero, x, zero, zero)

        assert_equal_with_reltol(potential0, potential, 8)

def test_particles_overlay(nbody_instance):
    """Test ParticlesOverlay functionality with N-body code."""
    particles = datamodel.Particles(
        mass=[1, 2] | nbody_system.mass,
        x=[-1, 1] | nbody_system.length,
        y=[-1, 1] | nbody_system.length,
        z=[-1, 1] | nbody_system.length,
        vx=[-1, 1] | nbody_system.speed,
        vy=[-1, 1] | nbody_system.speed,
        vz=[-1, 1] | nbody_system.speed
    )

    instance = nbody_instance

    overlay = datamodel.ParticlesOverlay(instance.particles)

    overlay.add_particles(particles)
    all_attributes = overlay.get_values_in_store(overlay.get_all_indices_in_store(), ['mass', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

    expected_mass = [1, 2] | nbody_system.mass
    assert_equal(all_attributes[0], expected_mass)
    assert_equal(instance.particles.mass, expected_mass)
    assert_equal(overlay.mass, expected_mass)
    expected_position = [[-1., -1., -1.], [1.,  1.,  1.]] | nbody_system.length
    assert_equal(overlay.position, expected_position)

