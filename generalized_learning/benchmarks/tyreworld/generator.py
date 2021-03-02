
import pathlib
import random
import shutil
import subprocess

from benchmarks.generator import Generator
from util import constants
from util import file


class TyreworldDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["domain_name", "min_tires", "max_tires"]) \
        .union(Generator.REQUIRED_KEYS)

    DOMAIN_FILE = (pathlib.Path(__file__).absolute().parent /
                   "domain.pddl").resolve()

    GENERATOR_BIN = (pathlib.Path(
        __file__).absolute().parent / "tyreworld").resolve()

    MIN_TIRES = 1

    _DOMAIN_NAME = "tyreworld"

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(TyreworldDomainGenerator, self).__init__(parent, parent_dir,
                                                       global_dict,
                                                       user_phase_dict,
                                                       failfast)

    def generate_domain(self):

        file_path = "%s/%s.domain.pddl" % (
            self._base_dir,
            TyreworldDomainGenerator._DOMAIN_NAME)

        shutil.copy(TyreworldDomainGenerator.DOMAIN_FILE,
                    file_path)

    def generate_problem(self, problem_range):

        min_tires = self.get_value("min_tires")
        max_tires = self.get_value("max_tires")

        assert min_tires >= TyreworldDomainGenerator.MIN_TIRES

        for problem_no in problem_range:

            problem_file = "problem_%u.problem.pddl" % (problem_no)
            file_handle = open("%s/%s" % (self._base_dir, problem_file), "w")

            tires = random.randint(min_tires, max_tires)

            properties = {

                "min_tires": min_tires,
                "max_tires": max_tires,
                "tires": tires,
            }

            file.write_properties(file_handle, properties,
                                  constants.PDDL_COMMENT_PREFIX)

            gen_cmd = "%s -n %u" % (
                TyreworldDomainGenerator.GENERATOR_BIN,
                tires)

            unused_completed_process = subprocess.run(
                gen_cmd, shell=True, stdout=file_handle)

        # Just return an empty list.
        return []
