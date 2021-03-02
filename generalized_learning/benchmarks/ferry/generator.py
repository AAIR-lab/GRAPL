
import pathlib
import random
import shutil
import subprocess
import sys

from benchmarks.generator import Generator
from util import constants
from util import file


class FerryDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["min_locations", "max_locations",
                         "min_cars", "max_cars"]) \
        .union(Generator.REQUIRED_KEYS)

    DOMAIN_FILE = (pathlib.Path(__file__).absolute().parent /
                   "domain.pddl").resolve()

    GENERATOR_BIN = (pathlib.Path(
        __file__).absolute().parent / "ferry").resolve()

    MIN_LOCATIONS = 2
    MIN_CARS = 1

    _DOMAIN_NAME = "ferry"
    MAX_SEED = (1 << sys.int_info.bits_per_digit) - 1

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(FerryDomainGenerator, self).__init__(parent, parent_dir,
                                                   global_dict,
                                                   user_phase_dict,
                                                   failfast)

    def generate_domain(self):

        file_path = "%s/%s.domain.pddl" % (
            self._base_dir,
            FerryDomainGenerator._DOMAIN_NAME)

        shutil.copy(FerryDomainGenerator.DOMAIN_FILE,
                    file_path)

    def generate_problem(self, problem_range):

        min_locations = self.get_value("min_locations")
        max_locations = self.get_value("max_locations")
        min_cars = self.get_value("min_cars")
        max_cars = self.get_value("max_cars")

        assert min_locations >= FerryDomainGenerator.MIN_LOCATIONS
        assert min_cars >= FerryDomainGenerator.MIN_CARS

        for problem_no in problem_range:

            problem_file = "problem_%u.problem.pddl" % (problem_no)
            file_handle = open("%s/%s" % (self._base_dir, problem_file), "w")

            locations = random.randint(min_locations, max_locations)
            cars = random.randint(min_cars, max_cars)
            seed = random.randint(1, FerryDomainGenerator.MAX_SEED)

            properties = {

                "min_locations": min_locations,
                "max_locations": max_locations,
                "min_cars": min_cars,
                "max_cars": max_cars,
                "locations": locations,
                "cars": cars,
                "seed": seed,

                "bin_params": ["locations", "cars"]
            }

            file.write_properties(file_handle, properties,
                                  constants.PDDL_COMMENT_PREFIX)

            gen_cmd = "%s -l %u -c %u -s %u" % (
                FerryDomainGenerator.GENERATOR_BIN,
                locations,
                cars,
                seed)

            unused_completed_process = subprocess.run(
                gen_cmd, shell=True, stdout=file_handle)

        # Just return an empty list.
        return []
