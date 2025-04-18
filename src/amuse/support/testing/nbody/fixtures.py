"""Fixtures for generic nbody tests.

They are **imported** by nbody_test.py.

NOTE: This means that any fixtures defined here take precedence over fixtures
in code-specific conftest.py. files. This behavior *differs* from pytest
behavior, where the fixture that is first discovered (in our case, the fixture
defined in the specific code) is used.
"""

from pytest import fixture
from amuse.units import nbody_system
from amuse.units import units
from amuse.units import quantities

from amuse import datamodel

from amuse.support import literature
literature.TrackLiteratureReferences.suppress_output()

# Factory to create nbody instances. Handles teardown for all
# Follows https://docs.pytest.org/en/stable/how-to/fixtures.html#factories-as-fixtures
@fixture()
def make_nbody_instance(nbody_implementation):
    """Create instance of an implementation of an nbody code.

    The function handles the teardown of the created object automatically.

    Args:
        nbody_implementation: Callable class that implements an nbody code.

    Returns:
        The created instance.
    """
    created_instances = []

    def _make_instance(*args, **kwargs):
        instance = nbody_implementation(*args, **kwargs)
        created_instances.append(instance)
        return instance

    yield _make_instance

    for instance in created_instances:
        instance.cleanup_code()
        instance.stop()


@fixture()
def nbody_instance(make_nbody_instance):
    """A single nbody instance fixture."""
    yield make_nbody_instance()


@fixture()
def nbody_instance_kg(make_nbody_instance):
    """A single nbody instance fixture for a kg-based nbody system."""
    convert_nbody = nbody_system.nbody_to_si(5.0 | units.kg, 10.0 | units.m)
    instance = make_nbody_instance(convert_nbody)
    yield instance

@fixture
def particle_fixture(request):
    """Create particles from input arguments.

    This fixture returns a set of `datamodel.Particles`with dynamic inputs.
    To use this particle fixture, first parametrize the test
    function with 2 arguments--this fixture and the particle inputs; declare
    the fixture as indirect. Then, pass the fixture as argument to the
    test function. See `nbody_tests.py` for examples.

    The particle inputs need to be a tuple. The first entry is an integer,
    denoting the number of particles. The second entry is a dictionary of particle
    attribute and value pairs.
    """
    num_particles, kwargs = request.param
    particles = datamodel.Particles(num_particles)
    for key, value in kwargs.items():
        setattr(particles, key, value)
    return particles
