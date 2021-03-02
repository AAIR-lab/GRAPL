
import pathlib
import random
import shutil
import subprocess

from benchmarks.generator import Generator
from util import constants
from util import file


class HanoiDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["domain_name", "min_pegs", "max_pegs"]) \
        .union(Generator.REQUIRED_KEYS)

    DOMAIN_FILE = (pathlib.Path(__file__).absolute().parent /
                   "domain.pddl").resolve()

    GENERATOR_BIN = (pathlib.Path(
        __file__).absolute().parent / "hanoi").resolve()

    MIN_PEGS = 1

    _DOMAIN_NAME = "hanoi"

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(HanoiDomainGenerator, self).__init__(parent, parent_dir,
                                                   global_dict,
                                                   user_phase_dict,
                                                   failfast)

    def generate_domain(self):

        file_path = "%s/%s.domain.pddl" % (
            self._base_dir,
            HanoiDomainGenerator._DOMAIN_NAME)

        shutil.copy(HanoiDomainGenerator.DOMAIN_FILE,
                    file_path)

    def generate_problem(self, problem_range):

        min_pegs = self.get_value("min_pegs")
        max_pegs = self.get_value("max_pegs")

        assert min_pegs >= HanoiDomainGenerator.MIN_PEGS

        for problem_no in problem_range:

            problem_file = "problem_%u.problem.pddl" % (problem_no)
            file_handle = open("%s/%s" % (self._base_dir, problem_file), "w")

            pegs = random.randint(min_pegs, max_pegs)

            properties = {

                "min_pegs": min_pegs,
                "max_pegs": max_pegs,
                "pegs": pegs,
            }

            file.write_properties(file_handle, properties,
                                  constants.PDDL_COMMENT_PREFIX)

            gen_cmd = "%s -n %u" % (
                HanoiDomainGenerator.GENERATOR_BIN,
                pegs)

            unused_completed_process = subprocess.run(
                gen_cmd, shell=True, stdout=file_handle)

        # Just return an empty list.
        return []
