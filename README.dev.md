
# AMUSE developer documentation

## Community codes

### Testing of `nbody` codes

(This idea will be extended to other community codes, but for now it is
only implemented for `nbody` codes.)

Each implementation of `nbody` codes, such as `bhtree`, ph4 and `hermite`, should
pass a set standard tests to ensure they implement the AMUSE gravitational dynamics
interface correctly ("API tests").

To run the API tests on a community code:
1. In a local `conftest.py` file, define the following fixtures:
    - `nbody_implementation`, which returns the class to be tested
    - `nbody_timestep_parameter`, which returns a tuple of name of the "timestep"
    parameter in this implementation, and the value in `nbody_system.time` units.
    - `starting_particle_index`: The starting index for retrieving particle
    attributes from a `nbody` instance.
2. In a local test file, such as `test_bhtree.py`, add this line of code:
    ```python
    from amuse.support.testing.nbody.nbody_tests import *
    ```

Invoking `pytest test_bhtree.py` will automatically run the API tests.

Besides the API tests, the community codes can also test functionality that is specific
to their implementation. This can be tested in the usual way: add the specific
tests to the community code as a normal test for `pytest`.

### Creating new API tests

There should be a single file in `src/amuse/support/testing/*` that contains
all test functions. The functions need to be named `test_*` so that they are
discovered by `pytest`.

**Fixtures**
- All fixtures that are not specific to a code should
be defined inside the AMUSE framework, and be imported into the file
with the test functions with `from file_with_fixtures import *`. Important: this
will override any fixtures defined in *any* `conftest` file.
- Any fixtures that are specific to the code should be defined in the local
code package in a `conftest.py` file. The standard `pytest` discovery
mechanism applies to them.













