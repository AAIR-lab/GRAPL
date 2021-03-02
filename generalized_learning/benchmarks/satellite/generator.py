
import pathlib
import random
import shutil
import subprocess
import sys

from benchmarks.generator import Generator
from util import constants
from util import file


class SatelliteDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["min_satellites", "max_satellites",
                         "min_instruments", "max_instruments",
                         "min_modes", "max_modes",
                         "min_targets", "max_targets",
                         "min_observations", "max_observations"]) \
        .union(Generator.REQUIRED_KEYS)

    DOMAIN_FILE = (pathlib.Path(__file__).absolute().parent /
                   "domain.pddl").resolve()

    GENERATOR_BIN = (pathlib.Path(
        __file__).absolute().parent / "satellite").resolve()

    MIN_SATELLITES = 1
    MIN_INSTRUMENTS = 1
    MIN_MODES = 1
    MIN_TARGETS = 1
    MIN_OBSERVATIONS = 1

    MAX_SEED = (1 << sys.int_info.bits_per_digit) - 1

    _DOMAIN_NAME = "satellite"

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(SatelliteDomainGenerator, self).__init__(parent, parent_dir,
                                                       global_dict,
                                                       user_phase_dict,
                                                       failfast)

    def generate_domain(self):

        file_path = "%s/%s.domain.pddl" % (
            self._base_dir,
            SatelliteDomainGenerator._DOMAIN_NAME)

        shutil.copy(SatelliteDomainGenerator.DOMAIN_FILE,
                    file_path)

    def generate_problem(self, problem_range):

        min_satellites = self.get_value("min_satellites")
        max_satellites = self.get_value("max_satellites")
        min_instruments = self.get_value("min_instruments")
        max_instruments = self.get_value("max_instruments")
        min_modes = self.get_value("min_modes")
        max_modes = self.get_value("max_modes")
        min_targets = self.get_value("min_targets")
        max_targets = self.get_value("max_targets")
        min_observations = self.get_value("min_observations")
        max_observations = self.get_value("max_observations")

        assert min_satellites >= SatelliteDomainGenerator.MIN_SATELLITES
        assert min_instruments >= SatelliteDomainGenerator.MIN_INSTRUMENTS
        assert min_modes >= SatelliteDomainGenerator.MIN_MODES
        assert min_targets >= SatelliteDomainGenerator.MIN_TARGETS
        assert min_observations >= SatelliteDomainGenerator.MIN_OBSERVATIONS

        for problem_no in problem_range:

            problem_file = "problem_%u.problem.pddl" % (problem_no)
            file_handle = open("%s/%s" % (self._base_dir, problem_file), "w")

            satellites = random.randint(min_satellites, max_satellites)
            instruments = random.randint(min_instruments, max_instruments)
            modes = random.randint(min_modes, max_modes)
            targets = random.randint(min_targets, max_targets)
            observations = random.randint(min_observations, max_observations)
            seed = random.randint(1, SatelliteDomainGenerator.MAX_SEED)

            properties = {

                "min_satellites": min_satellites,
                "max_satellites": max_satellites,
                "min_instruments": min_instruments,
                "max_instruments": max_instruments,
                "min_modes": min_modes,
                "max_modes": max_modes,
                "min_targets": min_targets,
                "max_targets": max_targets,
                "min_observations": min_observations,
                "max_observations": max_observations,

                "satellites": satellites,
                "instruments": instruments,
                "modes": modes,
                "targets": targets,
                "observations": observations,
                "seed": seed,

                "bin_params": ["satellites",
                               "instruments",
                               "modes",
                               "targets",
                               "observations"]
            }

            file.write_properties(file_handle, properties,
                                  constants.PDDL_COMMENT_PREFIX)

            gen_cmd = "%s %s %u %u %u %u %u" % (
                SatelliteDomainGenerator.GENERATOR_BIN,
                seed,
                satellites,
                instruments,
                modes,
                targets,
                observations)

            unused_completed_process = subprocess.run(
                gen_cmd, shell=True, stdout=file_handle)

        # Just return an empty list.
        return []
