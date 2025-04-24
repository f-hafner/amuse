from amuse.units import nbody_system
from amuse.units import units


list_new_particle = [
    15.0 | nbody_system.mass,
    10.0 | nbody_system.length,
    20.0 | nbody_system.length,
    30.0 | nbody_system.length,
    1.0 | nbody_system.speed,
    1.0 | nbody_system.speed,
    3.0 | nbody_system.speed,
    10.0 | nbody_system.length
    ]

list_new_particle_kg = [
    15.0 | units.kg,
    10.0 | units.m,
    20.0 | units.m,
    30.0 | units.m,
    0.0 | units.m/units.s,
    0.0 | units.m/units.s,
    0.0 | units.m/units.s,
    10.0 | units.m
    ]

kg_particle = (2, {"mass": [15.0, 30.0] | units.kg,
          "radius": [10.0, 20.0] | units.m,
          "position": [[10.0, 20.0, 30.0], [20.0, 40.0, 60.0]] | units.m,
          "velocity": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] | units.m / units.s
          })

center_of_mass_position = (2, {"mass": [30.0, 30.0] | units.kg,
          "radius": [1.0, 1.0] | units.m,
          "position": [[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]] | units.m,
          "velocity": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] | units.m / units.s
          })

collision_detection = (
        7, {"x": [-101.0, -100.0, -0.5, 0.5, 100.0, 101.0, 104.0] | nbody_system.length,
            "y": 0 | nbody_system.length,
            "z": 0 | nbody_system.length,
            "mass": 0.001 | nbody_system.mass,
            "radius": 0.01 | nbody_system.length,
            "velocity": [[2, 0, 0], [-2, 0, 0]]*3 + [[-4, 0, 0]] | nbody_system.speed
          })

stop_n_steps = (
        2, {"x": [0.0, 10.0] | nbody_system.length,
            "y": 0 | nbody_system.length,
            "z": 0 | nbody_system.length,
            "radius": 0.005 | nbody_system.length,
            "vx": 0 | nbody_system.speed,
            "vy": 0 | nbody_system.speed,
            "vz": 0 | nbody_system.speed,
            "mass": 1.0 | nbody_system.mass,
          })

direction_and_speed_when_evolving_model = (
        2, {"x": [0.0, 10.0] | nbody_system.length,
            "y": 0.0 | nbody_system.length,
            "z": 0.0 | nbody_system.length,
            "vx": 1.0 | nbody_system.speed,
            "vy": 0.0 | nbody_system.speed,
            "vz": 0.0 | nbody_system.speed,
            "mass": 0.1 | nbody_system.mass,
          })

gravity_with_same_potential = (
        2, {"mass": [1.0, 1.0] | nbody_system.mass,
            "radius": [0.0001, 0.0001] | nbody_system.length,
            "position": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]] | nbody_system.length,
            "velocity": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] | nbody_system.speed
          })

gravity_at_positions = (
        6, {"mass": 1.0 | nbody_system.mass,
            "radius": 0.0001 | nbody_system.length,
            "position": [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]] | nbody_system.length,
            "velocity": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] | nbody_system.speed
          })
