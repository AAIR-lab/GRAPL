
import pathlib
import random
import shutil
import subprocess
import sys

from benchmarks.generator import Generator

from util import constants
from util import file
from util.util import round_no


class ChildsnackDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["min_children", "max_children",
                         "min_trays", "max_trays",
                         "min_gluten_ratio", "max_gluten_ratio",
                         "min_sandwich_ratio", "max_sandwich_ratio"]) \
        .union(Generator.REQUIRED_KEYS)

    DOMAIN_FILE = (pathlib.Path(__file__).absolute().parent /
                   "domain.pddl").resolve()

    GENERATOR_BIN = (pathlib.Path(
        __file__).absolute().parent / "childsnack.py").resolve()

    MAX_SEED = (1 << sys.int_info.bits_per_digit) - 1
    PROBLEM_TYPE = "pool"
    MIN_CHILDREN = 1
    MIN_TRAYS = 1

    # Always have # of sandwiches be >= # of children to guarantee solvability.
    MIN_SANDWICH_RATIO = 1.0
    MIN_ROWS = 2

    _DOMAIN_NAME = "childsnack"

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(ChildsnackDomainGenerator, self).__init__(parent, parent_dir,
                                                        global_dict,
                                                        user_phase_dict,
                                                        failfast)

    def generate_domain(self):

        file_path = "%s/%s.domain.pddl" % (
            self._base_dir,
            ChildsnackDomainGenerator._DOMAIN_NAME)

        shutil.copy(ChildsnackDomainGenerator.DOMAIN_FILE,
                    file_path)

    def generate_problem(self, problem_range):

        min_children = self.get_value("min_children")
        max_children = self.get_value("max_children")
        min_trays = self.get_value("min_trays")
        max_trays = self.get_value("max_trays")
        min_gluten_ratio = self.get_value("min_gluten_ratio")
        max_gluten_ratio = self.get_value("max_gluten_ratio")
        min_sandwich_ratio = self.get_value("min_sandwich_ratio")
        max_sandwich_ratio = self.get_value("max_sandwich_ratio")

        assert min_children >= ChildsnackDomainGenerator.MIN_CHILDREN
        assert min_trays >= ChildsnackDomainGenerator.MIN_TRAYS
        assert min_sandwich_ratio >= \
            ChildsnackDomainGenerator.MIN_SANDWICH_RATIO

        for problem_no in problem_range:

            problem_file = "problem_%u.problem.pddl" % (problem_no)
            file_handle = open("%s/%s" % (self._base_dir, problem_file), "w")

            seed = random.randint(0, ChildsnackDomainGenerator.MAX_SEED)
            children = random.randint(min_children, max_children)
            trays = random.randint(min_trays, max_trays)
            gluten_ratio = random.uniform(min_gluten_ratio, max_gluten_ratio)

            gluten_ratio = round_no(gluten_ratio, prec=2, base=0.25)
            sandwich_ratio = random.uniform(min_sandwich_ratio,
                                            max_sandwich_ratio)

            properties = {

                "min_children": min_children,
                "max_children": max_children,
                "min_trays": min_trays,
                "max_trays": max_trays,
                "min_gluten_ratio": min_gluten_ratio,
                "max_gluten_ratio": max_gluten_ratio,
                "min_sandwich_ratio": min_sandwich_ratio,
                "max_sandwich_ratio": max_sandwich_ratio,
                "seed": seed,
                "children": children,
                "trays": trays,
                "gluten_ratio": "%.1f" % (gluten_ratio),
                "sandwich_ratio": "%.1f" % (sandwich_ratio),

                "bin_params": ["children", "trays"]
            }

            file.write_properties(file_handle, properties,
                                  constants.PDDL_COMMENT_PREFIX)

            # Use %.1f to force the 2nd endpoint of the ratio values (the max).
            # example: %.1f(1.9xxx...) = 2.0
            gen_cmd = "%s %s %u %u %u %.1f %.1f" % (
                ChildsnackDomainGenerator.GENERATOR_BIN,
                ChildsnackDomainGenerator.PROBLEM_TYPE,
                seed,
                children,
                trays,
                gluten_ratio,
                sandwich_ratio)

            unused_completed_process = subprocess.run(
                gen_cmd, shell=True, stdout=file_handle)

        # Just return an empty list.
        return []
