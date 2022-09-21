
import pathlib
import random
import shutil
import subprocess

from benchmarks.generator import Generator
from generalized_learning.concretized.problem import Problem

from util import constants
from util import file


class GripperDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["min_balls", "max_balls"]) \
        .union(Generator.REQUIRED_KEYS)

    DOMAIN_FILE = (pathlib.Path(__file__).absolute().parent /
                   "domain.pddl").resolve()

    GENERATOR_BIN = (pathlib.Path(
        __file__).absolute().parent / "gripper").resolve()

    MIN_BALLS = 1

    _DOMAIN_NAME = "gripper"

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(GripperDomainGenerator, self).__init__(parent, parent_dir,
                                                     global_dict,
                                                     user_phase_dict,
                                                     failfast)

    def generate_domain(self):

        file_path = "%s/%s.domain.pddl" % (
            self._base_dir,
            GripperDomainGenerator._DOMAIN_NAME)

        shutil.copy(GripperDomainGenerator.DOMAIN_FILE,
                    file_path)

    def _generate_problem(self, domain_file, problem_file,
                          min_balls, max_balls, balls):

        file_handle = open("%s/%s" % (self._base_dir, problem_file), "w")

        properties = {

            "min_balls": min_balls,
            "max_balls": max_balls,
            "balls": balls,

            "bin_params": ["balls"],
        }

        file.write_properties(file_handle, properties,
                              constants.PDDL_COMMENT_PREFIX)

        gen_cmd = "%s -n %u" % (
            GripperDomainGenerator.GENERATOR_BIN,
            balls)

        unused_completed_process = subprocess.run(
            gen_cmd, shell=True, stdout=file_handle)

        file_handle.close()

        problem = Problem(domain_file, problem_file, directory=self._base_dir)
        return problem.requires_planning()

    def generate_problem(self, problem_range):

        min_balls = self.get_value("min_balls")
        max_balls = self.get_value("max_balls")

        assert min_balls >= GripperDomainGenerator.MIN_BALLS

        domain_file = "%s.domain.pddl" % (
            GripperDomainGenerator._DOMAIN_NAME)

        for problem_no in problem_range:

            problem_file = "problem_%u.problem.pddl" % (problem_no)

            balls = random.randint(min_balls, max_balls)

            i = 0
            success = False
            while i < Generator.MAX_TRIES and not success:

                i += 1

                success |= self._generate_problem(domain_file, problem_file,
                                                  min_balls, max_balls,
                                                  balls)

            if not success:

                raise Exception("Could not generate problem")

        # Just return an empty list.
        return []
