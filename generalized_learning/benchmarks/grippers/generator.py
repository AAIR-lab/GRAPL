
import pathlib
import random
import shutil
import subprocess

from benchmarks.generator import Generator
from util import constants
from util import file


class GrippersDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["min_robots", "max_robots",
                         "min_rooms", "max_rooms",
                         "min_balls", "max_balls"]) \
        .union(Generator.REQUIRED_KEYS)

    DOMAIN_FILE = (pathlib.Path(__file__).absolute().parent /
                   "domain.pddl").resolve()

    GENERATOR_BIN = (pathlib.Path(
        __file__).absolute().parent / "grippers").resolve()

    MIN_ROBOTS = 1
    MIN_ROOMS = 1
    MIN_BALLS = 1

    _DOMAIN_NAME = "grippers"

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(GrippersDomainGenerator, self).__init__(parent, parent_dir,
                                                      global_dict,
                                                      user_phase_dict,
                                                      failfast)

    def generate_domain(self):

        file_path = "%s/%s.domain.pddl" % (
            self._base_dir,
            GrippersDomainGenerator._DOMAIN_NAME)

        shutil.copy(GrippersDomainGenerator.DOMAIN_FILE,
                    file_path)

    def generate_problem(self, problem_range):

        min_robots = self.get_value("min_robots")
        max_robots = self.get_value("max_robots")
        min_rooms = self.get_value("min_rooms")
        max_rooms = self.get_value("max_rooms")
        min_balls = self.get_value("min_balls")
        max_balls = self.get_value("max_balls")

        assert min_robots >= GrippersDomainGenerator.MIN_ROBOTS
        assert min_rooms >= GrippersDomainGenerator.MIN_ROOMS
        assert min_balls >= GrippersDomainGenerator.MIN_BALLS

        for problem_no in problem_range:

            problem_file = "problem_%u.problem.pddl" % (problem_no)
            file_handle = open("%s/%s" % (self._base_dir, problem_file), "w")

            robots = random.randint(min_robots, max_robots)
            rooms = random.randint(min_rooms, max_rooms)
            balls = random.randint(min_balls, max_balls)

            properties = {

                "min_robots": min_robots,
                "max_robots": max_robots,
                "min_rooms": min_rooms,
                "max_rooms": max_rooms,
                "min_balls": min_balls,
                "max_balls": max_balls,
                "robots": robots,
                "rooms": rooms,
                "balls": balls,

                "bin_params": ["robots", "rooms", "balls"]
            }

            file.write_properties(file_handle, properties,
                                  constants.PDDL_COMMENT_PREFIX)

            gen_cmd = "%s -n %u -r %u -o %u" % (
                GrippersDomainGenerator.GENERATOR_BIN,
                robots,
                rooms,
                balls)

            unused_completed_process = subprocess.run(
                gen_cmd, shell=True, stdout=file_handle)

        # Just return an empty list.
        return []
