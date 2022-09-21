
import pathlib
import random
import shutil
import subprocess

from benchmarks.generator import Generator
from generalized_learning.concretized.problem import Problem

from util import constants
from util import file


class MiconicDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["min_floors", "max_floors",
                         "min_passengers", "max_passengers"]) \
        .union(Generator.REQUIRED_KEYS)

    DOMAIN_FILE = (pathlib.Path(__file__).absolute().parent /
                   "domain.pddl").resolve()

    GENERATOR_BIN = (pathlib.Path(
        __file__).absolute().parent / "miconic").resolve()

    MIN_FLOORS = 2
    MIN_PASSENGERS = 1

    _DOMAIN_NAME = "miconic"

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(MiconicDomainGenerator, self).__init__(parent, parent_dir,
                                                     global_dict,
                                                     user_phase_dict,
                                                     failfast)

    def generate_domain(self):

        file_path = "%s/%s.domain.pddl" % (
            self._base_dir,
            MiconicDomainGenerator._DOMAIN_NAME)

        shutil.copy(MiconicDomainGenerator.DOMAIN_FILE,
                    file_path)

    def _generate_problem(self, domain_file, problem_file,
                          min_floors, max_floors,
                          min_passengers, max_passengers,
                          floors, passengers):

        file_handle = open("%s/%s" % (self._base_dir, problem_file), "w")

        properties = {

            "min_floors": min_floors,
            "max_floors": max_floors,
            "min_passengers": min_passengers,
            "max_passengers": max_passengers,
            "floors": floors,
            "passengers": passengers,

            "bin_params": ["floors", "passengers"]
        }

        file.write_properties(file_handle, properties,
                              constants.PDDL_COMMENT_PREFIX)

        gen_cmd = "%s -f %u -p %u" % (MiconicDomainGenerator.GENERATOR_BIN,
                                      floors,
                                      passengers)

        unused_completed_process = subprocess.run(
            gen_cmd, shell=True, stdout=file_handle)

        file_handle.close()

        problem = Problem(domain_file, problem_file, directory=self._base_dir)
        return problem.requires_planning()

    def generate_problem(self, problem_range):

        min_floors = self.get_value("min_floors")
        max_floors = self.get_value("max_floors")
        min_passengers = self.get_value("min_passengers")
        max_passengers = self.get_value("max_passengers")

        assert min_floors >= MiconicDomainGenerator.MIN_FLOORS
        assert min_passengers >= MiconicDomainGenerator.MIN_PASSENGERS

        domain_file = "%s.domain.pddl" % (
            MiconicDomainGenerator._DOMAIN_NAME)

        for problem_no in problem_range:

            problem_file = "problem_%u.problem.pddl" % (problem_no)

            floors = random.randint(min_floors, max_floors)
            passengers = random.randint(min_passengers, max_passengers)

            i = 0
            success = False
            while i < Generator.MAX_TRIES and not success:

                i += 1

                success |= self._generate_problem(domain_file, problem_file,
                                                  min_floors, max_floors,
                                                  min_passengers, max_passengers,
                                                  floors, passengers)

            if not success:

                raise Exception("Could not generate problem")

        # Just return an empty list.
        return []
