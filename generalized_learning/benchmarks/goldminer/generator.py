
import pathlib
import random
import shutil
import subprocess
import sys

from benchmarks.generator import Generator
from util import constants
from util import file


class GoldminerDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["min_rows", "max_rows",
                         "min_columns", "max_columns"]) \
        .union(Generator.REQUIRED_KEYS)

    DOMAIN_FILE = (pathlib.Path(__file__).absolute().parent /
                   "domain.pddl").resolve()

    GENERATOR_BIN = (pathlib.Path(
        __file__).absolute().parent / "goldminer").resolve()

    MIN_ROWS = 2
    MIN_COLUMNS = 2

    _DOMAIN_NAME = "goldminer"
    MAX_SEED = (1 << sys.int_info.bits_per_digit) - 1

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(GoldminerDomainGenerator, self).__init__(parent, parent_dir,
                                                       global_dict,
                                                       user_phase_dict,
                                                       failfast)

    def generate_domain(self):

        file_path = "%s/%s.domain.pddl" % (
            self._base_dir,
            GoldminerDomainGenerator._DOMAIN_NAME)

        shutil.copy(GoldminerDomainGenerator.DOMAIN_FILE,
                    file_path)

    def generate_problem(self, problem_range):

        min_rows = self.get_value("min_rows")
        max_rows = self.get_value("max_rows")
        min_columns = self.get_value("min_columns")
        max_columns = self.get_value("max_columns")

        assert min_rows >= GoldminerDomainGenerator.MIN_ROWS
        assert min_columns >= GoldminerDomainGenerator.MIN_COLUMNS

        for problem_no in problem_range:

            problem_file = "problem_%u.problem.pddl" % (problem_no)
            file_handle = open("%s/%s" % (self._base_dir, problem_file), "w")

            rows = random.randint(min_rows, max_rows)
            columns = random.randint(min_columns, max_columns)
            seed = random.randint(1, GoldminerDomainGenerator.MAX_SEED)
            properties = {

                "min_rows": min_rows,
                "max_rows": max_rows,
                "min_columns": min_columns,
                "max_columns": max_columns,
                "rows": rows,
                "columns": columns,
                "seed": seed,

                "bin_params": ["rows", "columns"]
            }

            file.write_properties(file_handle, properties,
                                  constants.PDDL_COMMENT_PREFIX)

            gen_cmd = "%s -r %u -c %u -s %u" % (
                GoldminerDomainGenerator.GENERATOR_BIN,
                rows,
                columns,
                seed)

            unused_completed_process = subprocess.run(
                gen_cmd, shell=True, stdout=file_handle)

        # Just return an empty list.
        return []
