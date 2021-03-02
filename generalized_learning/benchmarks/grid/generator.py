
import pathlib
import random
import shutil
import subprocess

from benchmarks.generator import Generator
from util import constants
from util import file


class GridDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["min_x", "max_x",
                         "min_y", "max_y",
                         "min_types", "max_types",
                         "min_keys", "max_keys",
                         "min_locks", "max_locks",
                         "min_prob", "max_prob"]) \
        .union(Generator.REQUIRED_KEYS)

    DOMAIN_FILE = (pathlib.Path(__file__).absolute().parent /
                   "domain.pddl").resolve()

    GENERATOR_BIN = (pathlib.Path(
        __file__).absolute().parent / "grid").resolve()

    MIN_X = 2
    MIN_Y = 2
    MIN_TYPES = 1
    MIN_KEYS = 1
    MIN_LOCKS = 1

    MIN_PROBABILITY = 0
    MAX_PROBABILITY = 100

    _DOMAIN_NAME = "grid"

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(GridDomainGenerator, self).__init__(parent, parent_dir,
                                                  global_dict,
                                                  user_phase_dict,
                                                  failfast)

    def generate_domain(self):

        file_path = "%s/%s.domain.pddl" % (
            self._base_dir,
            GridDomainGenerator._DOMAIN_NAME)

        shutil.copy(GridDomainGenerator.DOMAIN_FILE,
                    file_path)

    def generate_problem(self, problem_range):

        min_x = self.get_value("min_x")
        max_x = self.get_value("max_x")
        min_y = self.get_value("min_y")
        max_y = self.get_value("max_y")
        min_types = self.get_value("min_types")
        max_types = self.get_value("max_types")
        min_keys = self.get_value("min_keys")
        max_keys = self.get_value("max_keys")
        min_locks = self.get_value("min_locks")
        max_locks = self.get_value("max_locks")
        min_prob = self.get_value("min_prob")
        max_prob = self.get_value("max_prob")

        assert min_x >= GridDomainGenerator.MIN_X
        assert min_y >= GridDomainGenerator.MIN_Y
        assert min_types >= GridDomainGenerator.MIN_TYPES
        assert min_keys >= GridDomainGenerator.MIN_KEYS
        assert min_locks >= GridDomainGenerator.MIN_LOCKS
        assert min_prob >= GridDomainGenerator.MIN_PROBABILITY
        assert max_prob <= GridDomainGenerator.MAX_PROBABILITY

        for problem_no in problem_range:

            problem_file = "problem_%u.problem.pddl" % (problem_no)
            file_handle = open("%s/%s" % (self._base_dir, problem_file), "w")

            x = random.randint(min_x, max_x)
            y = random.randint(min_y, max_y)
            types = random.randint(min_types, max_types)
            keys = random.randint(min_keys, max_keys)
            locks = random.randint(min_locks, max_locks)
            probability = random.randint(min_prob, max_prob)

            # Legal values for keys and locks.
            keys = min((x * y) - 2, keys)
            locks = min(keys, locks)
            locks = max(1, locks)

            properties = {

                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "min_types": min_types,
                "max_types": max_types,
                "min_keys": min_keys,
                "max_keys": max_keys,
                "min_locks": min_locks,
                "max_locks": max_locks,
                "min_prob": min_prob,
                "max_prob": max_prob,
                "x": x,
                "y": y,
                "types": types,
                "keys": keys,
                "locks": locks,
                "probability": probability,

                "bin_params": ["x", "y", "keys", "locks"],
            }

            file.write_properties(file_handle, properties,
                                  constants.PDDL_COMMENT_PREFIX)

            gen_cmd = "%s -x %u -y %u -t %u -k %u -l %u -p %u" % (
                GridDomainGenerator.GENERATOR_BIN,
                x,
                y,
                types,
                keys,
                locks,
                probability)

            unused_completed_process = subprocess.run(
                gen_cmd, shell=True, stdout=file_handle)

        # Just return an empty list.
        return []
