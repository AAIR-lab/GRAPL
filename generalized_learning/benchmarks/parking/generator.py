
import pathlib
import random
import shutil
import subprocess

from benchmarks.generator import Generator
from util import constants
from util import file


class ParkingDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["domain_name", "min_curbs", "max_curbs"]) \
        .union(Generator.REQUIRED_KEYS)

    DOMAIN_FILE = (pathlib.Path(__file__).absolute().parent /
                   "domain.pddl").resolve()

    GENERATOR_BIN = (pathlib.Path(
        __file__).absolute().parent / "parking.pl").resolve()

    MIN_CURBS = 2
    _MIN_CARS = 1

    _DOMAIN_NAME = "parking"

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(ParkingDomainGenerator, self).__init__(parent, parent_dir,
                                                     global_dict,
                                                     user_phase_dict,
                                                     failfast)

    def generate_domain(self):

        file_path = "%s/%s.domain.pddl" % (
            self._base_dir,
            ParkingDomainGenerator._DOMAIN_NAME)

        shutil.copy(ParkingDomainGenerator.DOMAIN_FILE,
                    file_path)

    def generate_problem(self, problem_range):

        min_curbs = self.get_value("min_curbs")
        max_curbs = self.get_value("max_curbs")

        assert min_curbs >= ParkingDomainGenerator.MIN_CURBS

        for problem_no in problem_range:

            problem_file = "problem_%u.problem.pddl" % (problem_no)
            file_handle = open("%s/%s" % (self._base_dir, problem_file), "w")

            curbs = random.randint(min_curbs, max_curbs)
            max_cars = 2 * curbs - 2
            cars = random.randint(ParkingDomainGenerator._MIN_CARS,
                                  max_cars)

            properties = {

                "min_curbs": min_curbs,
                "max_curbs": max_curbs,
                "min_cars": 1,
                "max_cars": max_cars,
                "curbs": curbs,
                "cars": cars,
            }

            file.write_properties(file_handle, properties,
                                  constants.PDDL_COMMENT_PREFIX)

            gen_cmd = "%s %u %u" % (
                ParkingDomainGenerator.GENERATOR_BIN,
                curbs,
                cars)

            unused_completed_process = subprocess.run(
                gen_cmd, shell=True, stdout=file_handle)

        # Just return an empty list.
        return []
