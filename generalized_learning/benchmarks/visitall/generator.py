
import pathlib
import random
import shutil
import subprocess
import sys

from benchmarks.generator import Generator
from util import constants
from util import file


class VisitAllDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["min_size", "max_size",
                         "min_g_percent", "max_g_percent", "min_hole_percent",
                         "max_hole_percent"]) \
        .union(Generator.REQUIRED_KEYS)

    DOMAIN_FILE = (pathlib.Path(__file__).absolute().parent /
                   "domain.pddl").resolve()

    GENERATOR_BIN = (pathlib.Path(
        __file__).absolute().parent / "visitall").resolve()

    MIN_SIZE = 2
    MIN_GOALS = 2

    _DOMAIN_NAME = "visitall"
    MAX_SEED = (1 << sys.int_info.bits_per_digit) - 1

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(VisitAllDomainGenerator, self).__init__(parent, parent_dir,
                                                      global_dict,
                                                      user_phase_dict,
                                                      failfast)

    def generate_domain(self):

        file_path = "%s/%s.domain.pddl" % (
            self._base_dir,
            VisitAllDomainGenerator._DOMAIN_NAME)

        shutil.copy(VisitAllDomainGenerator.DOMAIN_FILE,
                    file_path)

    def generate_problem(self, problem_range):

        min_size = self.get_value("min_size")
        max_size = self.get_value("max_size")
        min_g_percent = self.get_value("min_g_percent")
        max_g_percent = self.get_value("max_g_percent")
        min_hole_percent = self.get_value("min_hole_percent")
        max_hole_percent = self.get_value("max_hole_percent")

        assert min_size >= VisitAllDomainGenerator.MIN_SIZE
        assert min_g_percent >= 0 and max_g_percent <= 100
        assert min_hole_percent >= 0 and max_hole_percent <= 100

        for problem_no in problem_range:

            problem_file = "problem_%u.problem.pddl" % (problem_no)
            file_handle = open("%s/%s" % (self._base_dir, problem_file), "w")

            size = random.randint(min_size, max_size)
            goal_percent = random.randint(min_g_percent, max_g_percent)
            goals = int((size * size * goal_percent) / 100)
            goals = max(VisitAllDomainGenerator.MIN_GOALS, goals)

            hole_percent = random.randint(min_hole_percent, max_hole_percent)
            holes = int((((size * size) - goals - 1) * hole_percent) / 100)
            seed = random.randint(1, VisitAllDomainGenerator.MAX_SEED)

            properties = {

                "min_size": min_size,
                "max_size": max_size,
                "min_g_percent": min_g_percent,
                "max_g_percent": max_g_percent,
                "min_hole_percent": min_hole_percent,
                "max_hole_percent": max_hole_percent,
                "size": size,
                "goals": goals,
                "holes": holes,
                "seed": seed,

                "bin_params": ["size"]
            }

            file.write_properties(file_handle, properties,
                                  constants.PDDL_COMMENT_PREFIX)

            gen_cmd = "%s -n %u -r %.2f -u %u -s %u" % (
                VisitAllDomainGenerator.GENERATOR_BIN,
                size,
                goal_percent / 100.0,
                holes,
                seed)

            unused_completed_process = subprocess.run(
                gen_cmd, shell=True, stdout=file_handle)

        # Just return an empty list.
        return []
