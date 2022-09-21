
import pathlib
import random
import shutil
import subprocess

from benchmarks.generator import Generator
from generalized_learning.concretized.problem import Problem

from util import constants
from util import file


class BlocksworldDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["min_blocks", "max_blocks", "goal_type"]) \
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

    def write_tower(self, file_handle, tower):

        file_handle.write(" (clear %s)" % (tower[0]))

        for i in range(1, len(tower)):

            file_handle.write(" (on %s %s)" % (tower[i - 1], tower[i]))

        file_handle.write(" (on-table %s)" % (tower[len(tower) - 1]))

    def write_problem_header(self, file_handle, total_blocks):

        file_handle.write("(define (problem BW-rand-%u)\n" % (total_blocks))
        file_handle.write("(:domain blocksworld)\n")

    def write_problem_end(self, file_handle):

        file_handle.write(")\n")

    def write_problem_objects(self, file_handle, blocks):

        # Objects
        file_handle.write("(:objects")
        for block in blocks:

            file_handle.write(" %s" % (block))
        file_handle.write(")\n")

    def write_problem_initial_state(self, file_handle, initial_towers):

        file_handle.write("(:init (arm-empty)")
        for tower in initial_towers:

            self.write_tower(file_handle, tower)

        file_handle.write(")\n")

    def generate_objects(self, num_blocks):

        blocks = []
        for i in range(num_blocks):

            blocks.append("b%u" % (i))

        return blocks

    def generate_random_towers(self, blocks):

        towers = []
        remaining_blocks = set(blocks)
        while len(remaining_blocks) > 0:

            blocks_in_tower = random.randint(1, len(remaining_blocks))

            random_tower = random.sample(remaining_blocks, blocks_in_tower)

            towers.append(random_tower)

            remaining_blocks -= set(random_tower)

        return towers

    def generate_stack_all_goal(self, file_handle, num_blocks):

        blocks = self.generate_objects(num_blocks)

        initial_towers = self.generate_random_towers(blocks)

        goal_towers = []
        goal_towers.append(random.sample(blocks, num_blocks))

        self.write_problem_header(file_handle, len(blocks))
        self.write_problem_objects(file_handle, blocks)
        self.write_problem_initial_state(file_handle, initial_towers)

        file_handle.write("(:goal (and")
        for tower in goal_towers:

            for i in range(1, len(tower)):

                file_handle.write(" (on %s %s)" % (tower[i - 1], tower[i]))

        file_handle.write("))\n")

        self.write_problem_end(file_handle)

    def generate_on_A_B_goal(self, file_handle, num_blocks):

        blocks = self.generate_objects(num_blocks)

        initial_towers = self.generate_random_towers(blocks)

        goal_config = random.sample(blocks, 2)

        self.write_problem_header(file_handle, len(blocks))
        self.write_problem_objects(file_handle, blocks)
        self.write_problem_initial_state(file_handle, initial_towers)

        file_handle.write("(:goal (and (on %s %s)))\n" % (goal_config[0],
                                                          goal_config[1]))

        self.write_problem_end(file_handle)

    def _generate_problem(self, domain_file, problem_file, min_blocks,
                          max_blocks,
                          blocks, goal_type):

        file_handle = open("%s/%s" %
                           (self._base_dir, problem_file), "w")

        properties = {

            "min_blocks": min_blocks,
            "max_blocks": max_blocks,
            "blocks": blocks,

            "bin_params": ["blocks"]
        }

        file.write_properties(file_handle, properties,
                              constants.PDDL_COMMENT_PREFIX)

        if goal_type == "normal":

            gen_cmd = "%s -n %u" % (
                BlocksworldDomainGenerator.GENERATOR_BIN,
                blocks)

            unused_completed_process = subprocess.run(
                gen_cmd, shell=True, stdout=file_handle)
        elif goal_type == "stack_all":

            self.generate_stack_all_goal(file_handle, blocks)
        elif goal_type.lower() == "on_a_b":

            self.generate_on_A_B_goal(file_handle, blocks)
        else:

            raise Exception("Unknown goal type.")

        file_handle.close()

        problem = Problem(domain_file, problem_file, directory=self._base_dir)
        return problem.requires_planning()

    def generate_problem(self, problem_range):

        min_blocks = self.get_value("min_blocks")
        max_blocks = self.get_value("max_blocks")
        goal_type = self.get_value("goal_type")

        assert min_blocks >= BlocksworldDomainGenerator.MIN_BLOCKS

        domain_file = "%s.domain.pddl" % (
            BlocksworldDomainGenerator._DOMAIN_NAME)

        for problem_no in problem_range:

            problem_file = "problem_%u.problem.pddl" % (problem_no)

            blocks = random.randint(min_blocks, max_blocks)

            i = 0
            success = False
            while i < Generator.MAX_TRIES and not success:

                i += 1

                success |= self._generate_problem(domain_file, problem_file,
                                                  min_blocks, max_blocks,
                                                  blocks,
                                                  goal_type)

            if not success:

                raise Exception("Could not generate problem")

        # Just return an empty list.
        return []
