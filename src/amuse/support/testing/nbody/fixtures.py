
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
        nbody_implementation: Callable class that implements and nbody code.

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
    yield make_nbody_instance()


@fixture()
def nbody_instance_kg(make_nbody_instance): # for test4, test6, test7, test11, test12, test13
    convert_nbody = nbody_system.nbody_to_si(5.0 | units.kg, 10.0 | units.m)
    instance = make_nbody_instance(convert_nbody)
    # TODO: had to remove the below: fails on ph4
    #instance.commit_parameters()
    yield instance


@fixture()
def nbody_instance_with_particles(nbody_instance, particle_fixture):
    """Create and initialize an nbody instance once for each particle fixture."""
    nbody_instance.particles.add_particles(particle_fixture)
    nbody_instance.commit_particles()
    return nbody_instance



@fixture
def particle_fixture(request):
    """Create particles from input arguments."""
    num_particles, kwargs = request.param
    particles = datamodel.Particles(num_particles)
    for key, value in kwargs.items():
        setattr(particles, key, value)
    return particles




