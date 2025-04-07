from pytest import fixture
from amuse.community.bhtree.interface import BHTree

from amuse.units import nbody_system

from amuse.units import units
from amuse.units import quantities

@fixture()
def nbody_implementation():
    return BHTree

@fixture()
def nbody_timestep_parameter():
    return ("timestep", 0.00001 | nbody_system.time)

@fixture()
def starting_particle_index():
    """Return the starting index for particles in the BHTree code (1-based)."""
    return 1

