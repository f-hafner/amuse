from pytest import fixture

from amuse.community.hermite.interface import HermiteInterface, Hermite
from amuse.units import nbody_system
from amuse.units import units
from amuse.units import quantities

@fixture()
def nbody_implementation():
    return Hermite

@fixture()
def nbody_timestep_parameter():
    return ("dt_param", 0.01)

@fixture()
def starting_particle_index():
    """Return the starting index for particles in the Hermite code (0-based)."""
    return 0