
import pathlib
import random
import shutil
import subprocess

from benchmarks.generator import Generator

from util import constants
from util import file


class SokobanDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["min_size", "max_size",
                         "min_balls", "max_balls",
                         "min_walls", "max_walls"]) \
        .union(Generator.REQUIRED_KEYS)

    DOMAIN_FILE = (pathlib.Path(__file__).absolute().parent /
                   "domain.pddl").resolve()

    GENERATOR_BIN = (pathlib.Path(
        __file__).absolute().parent / "sokoban").resolve()

    MIN_SIZE = 5
    MIN_BALLS = 1
    MIN_WALLS = 0

    _DOMAIN_NAME = "sokoban"

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(SokobanDomainGenerator, self).__init__(parent, parent_dir,
                                                     global_dict,
                                                     user_phase_dict,
                                                     failfast)

    def generate_domain(self):

        file_path = "%s/%s.domain.pddl" % (
            self._base_dir,
            SokobanDomainGenerator._DOMAIN_NAME)

        shutil.copy(SokobanDomainGenerator.DOMAIN_FILE,
                    file_path)

    def generate_problem(self, problem_range):

        min_size = self.get_value("min_size")
        max_size = self.get_value("max_size")
        min_balls = self.get_value("min_balls")
        max_balls = self.get_value("max_balls")
        min_walls = self.get_value("min_walls")
        max_walls = self.get_value("max_walls")

        assert min_size >= SokobanDomainGenerator.MIN_SIZE
        assert min_balls >= SokobanDomainGenerator.MIN_BALLS
        assert min_walls >= SokobanDomainGenerator.MIN_WALLS

        for problem_no in problem_range:

            problem_file = "problem_%u.problem.pddl" % (problem_no)
            file_handle = open("%s/%s" % (self._base_dir, problem_file), "w")

            size = random.randint(min_size, max_size)
            balls = random.randint(min_balls, max_balls)
            walls = random.randint(min_walls, max_walls)

            properties = {

                "size": size,
                "min_size": min_size,
                "max_size": max_size,
                "min_balls": min_balls,
                "max_balls": max_balls,
                "min_walls": min_walls,
                "max_walls": max_walls,

                "size": size,
                "balls": balls,
                "walls": walls,

                "bin_params": ["size"]
            }

            file.write_properties(file_handle, properties,
                                  constants.PDDL_COMMENT_PREFIX)

            gen_cmd = "%s -n %u -b %u -w %u" % (
                SokobanDomainGenerator.GENERATOR_BIN,
                size,
                balls,
                walls)

            unused_completed_process = subprocess.run(
                gen_cmd, shell=True, stdout=file_handle)

        # Just return an empty list.
        return []
