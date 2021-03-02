
import pathlib
import random
import shutil
import subprocess

from benchmarks.generator import Generator

from util import constants
from util import file


class BlocksworldDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["min_blocks", "max_blocks"]) \
        .union(Generator.REQUIRED_KEYS)

    DOMAIN_FILE = (pathlib.Path(__file__).absolute().parent /
                   "domain.pddl").resolve()

    GENERATOR_BIN = (pathlib.Path(
        __file__).absolute().parent / "blocksworld").resolve()

    MIN_BLOCKS = 2

    _DOMAIN_NAME = "blocksworld"

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(BlocksworldDomainGenerator, self).__init__(parent, parent_dir,
                                                         global_dict,
                                                         user_phase_dict,
                                                         failfast)

    def generate_domain(self):

        file_path = "%s/%s.domain.pddl" % (
            self._base_dir,
            BlocksworldDomainGenerator._DOMAIN_NAME)

        shutil.copy(BlocksworldDomainGenerator.DOMAIN_FILE,
                    file_path)

    def generate_problem(self, problem_range):

        min_blocks = self.get_value("min_blocks")
        max_blocks = self.get_value("max_blocks")

        assert min_blocks >= BlocksworldDomainGenerator.MIN_BLOCKS

        for problem_no in problem_range:

            problem_file = "problem_%u.problem.pddl" % (problem_no)
            file_handle = open("%s/%s" % (self._base_dir, problem_file), "w")

            blocks = random.randint(min_blocks, max_blocks)

            properties = {

                "min_blocks": min_blocks,
                "max_blocks": max_blocks,
                "blocks": blocks,

                "bin_params": ["blocks"]
            }

            file.write_properties(file_handle, properties,
                                  constants.PDDL_COMMENT_PREFIX)

            gen_cmd = "%s -n %u" % (
                BlocksworldDomainGenerator.GENERATOR_BIN,
                blocks)

            unused_completed_process = subprocess.run(
                gen_cmd, shell=True, stdout=file_handle)

        # Just return an empty list.
        return []
