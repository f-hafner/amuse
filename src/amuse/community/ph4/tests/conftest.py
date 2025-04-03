
from pytest import fixture

from amuse.community.ph4.interface import ph4Interface, ph4
from amuse.units import nbody_system
from amuse.units import units
from amuse.units import quantities

@fixture()
def nbody_implementation():
    return ph4

@fixture()
def nbody_timestep_parameter():
    return ("timestep_parameter", 0.01)



