# -*- coding: utf-8 -*-
import numpy as np
from pytest import approx
from pytest import fixture
from amuse.community.bhtree.interface import BHTreeInterface

from amuse.support import literature
literature.TrackLiteratureReferences.suppress_output()

@fixture
def interface():
    interface = BHTreeInterface()
    interface.initialize_code()
    interface.commit_parameters()
    yield interface
    interface.cleanup_code()
    interface.stop()


def test_literature_references():
    instance = BHTreeInterface()
    assert "Barnes" in instance.all_literature_references_string()
    instance.stop()


def test_create_and_retrieve_particles(interface):
    res1 = interface.new_particle(mass=11.0, radius=2.0, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0)
    res2 = interface.new_particle(mass=21.0, radius=5.0, x=10.0, y=0.0, z=0.0, vx=10.0, vy=0.0, vz=0.0)

    interface.commit_particles()
    assert res1['index_of_the_particle'] == 1
    assert res2['index_of_the_particle'] == 2

    retrieved_state1 = interface.get_state(1)
    retrieved_state2 = interface.get_state(2)

    assert retrieved_state1["mass"] == 11.0
    assert retrieved_state2["mass"] == 21.0
    assert retrieved_state1["x"] == 0.0
    assert retrieved_state2["x"] == 10.0

    assert interface.get_index_of_first_particle()["index_of_the_particle"] == 1
    assert interface.get_index_of_next_particle(1)["index_of_the_next_particle"] == 2


def test_delete_particle_and_index_management(interface):
    for i in [1, 2, 3]:
        temp_particle = interface.new_particle(mass=i, radius=1.0, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0)
        assert temp_particle["index_of_the_particle"] == i

    interface.commit_particles()
    assert interface.get_index_of_first_particle()["index_of_the_particle"] == 1
    assert interface.get_index_of_next_particle(1)["index_of_the_next_particle"] == 2
    assert interface.get_index_of_next_particle(2)["index_of_the_next_particle"] == 3

    interface.delete_particle(1)

    assert interface.get_number_of_particles()["number_of_particles"] == 2

    # the deletion does a swap, so 3 is copied to 1, (overwriting old 1 and treesize -> treesize-1
    assert interface.get_index_of_first_particle()["index_of_the_particle"] == 3
    assert interface.get_index_of_next_particle(2)["__result"] == 1



def test_create_multiple_particles_at_once(interface):
    interface.new_particle([10, 20], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1])
    interface.commit_particles()
    retrieved_state = interface.get_state(1)

    assert retrieved_state["mass"] == 10.0
    assert retrieved_state["radius"] == 1

    retrieved_state = interface.get_state([1, 2])
    assert retrieved_state["mass"][1] == 20.0
    # TODO: we could test for return type (ordered dict), for keys of the returned object
    # Also, we could test for the entire data in mass (the entire list, not a single element)
    assert interface.get_number_of_particles()["number_of_particles"] == 2


def test_particle_id_after_delete_and_new(interface):
    ids = []
    for i in [1, 2, 3]:
        id, _ = interface.new_particle(mass=i, radius=1.0, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0)
        ids.append(id)

    interface.commit_particles()

    interface.delete_particle(ids[0])
    id, _ = interface.new_particle(mass=4, radius=1.0, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0)
    assert id != ids[-1]


def test_calculate_gravitational_potential(interface):
    interface.set_epsilon_squared(0.1 * 0.1)
    interface.commit_parameters()
    id1, errorcode = interface.new_particle(mass=10.0, radius=1.0, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0)
    id2, errorcode = interface.new_particle(mass=10.0, radius=1.0, x=2.0, y=0.0, z=0.0, vx=10.0, vy=0.0, vz=0.0)

    interface.commit_particles()
    potential, errorcode = interface.get_potential(id1)
    assert errorcode == 0

    # NOTE: -8 is taken from old code; it is semi-automated there
    assert potential == approx(-10.0 / np.sqrt(2.0**2 + 0.1**2), rel=1e-8)

