
import pathlib
import random
import shutil
import subprocess
import sys

from benchmarks.generator import Generator
from util import constants
from util import file


class DepotsDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["min_depots", "max_depots",
                         "min_distributors", "max_distributors",
                         "min_trucks", "max_trucks",
                         "min_pallets", "max_pallets",
                         "min_hoists", "max_hoists",
                         "min_crates", "max_crates"]) \
        .union(Generator.REQUIRED_KEYS)

    DOMAIN_FILE = (pathlib.Path(__file__).absolute().parent /
                   "domain.pddl").resolve()

    GENERATOR_BIN = (pathlib.Path(
        __file__).absolute().parent / "depots").resolve()

    MIN_DEPOTS = 1
    MIN_DISTRIBUTORS = 1
    MIN_TRUCKS = 1
    MIN_PALLETS = 1
    MIN_HOISTS = 1
    MIN_CRATES = 1

    MAX_SEED = (1 << sys.int_info.bits_per_digit) - 1

    _DOMAIN_NAME = "depots"

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(DepotsDomainGenerator, self).__init__(parent, parent_dir,
                                                    global_dict,
                                                    user_phase_dict,
                                                    failfast)

    def generate_domain(self):

        file_path = "%s/%s.domain.pddl" % (
            self._base_dir,
            DepotsDomainGenerator._DOMAIN_NAME)

        shutil.copy(DepotsDomainGenerator.DOMAIN_FILE,
                    file_path)

    def generate_problem(self, problem_range):

        min_depots = self.get_value("min_depots")
        max_depots = self.get_value("max_depots")
        min_distributors = self.get_value("min_distributors")
        max_distributors = self.get_value("max_distributors")
        min_trucks = self.get_value("min_trucks")
        max_trucks = self.get_value("max_trucks")
        min_pallets = self.get_value("min_pallets")
        max_pallets = self.get_value("max_pallets")
        min_hoists = self.get_value("min_hoists")
        max_hoists = self.get_value("max_hoists")
        min_crates = self.get_value("min_crates")
        max_crates = self.get_value("max_crates")

        assert min_depots >= DepotsDomainGenerator.MIN_DEPOTS
        assert min_distributors >= DepotsDomainGenerator.MIN_DISTRIBUTORS
        assert min_trucks >= DepotsDomainGenerator.MIN_TRUCKS
        assert min_pallets >= DepotsDomainGenerator.MIN_PALLETS
        assert min_hoists >= DepotsDomainGenerator.MIN_HOISTS
        assert min_crates >= DepotsDomainGenerator.MIN_CRATES

        for problem_no in problem_range:

            problem_file = "problem_%u.problem.pddl" % (problem_no)
            file_handle = open("%s/%s" % (self._base_dir, problem_file), "w")

            depots = random.randint(min_depots, max_depots)
            distributors = random.randint(min_distributors, max_distributors)
            trucks = random.randint(min_trucks, max_trucks)
            pallets = random.randint(min_pallets, max_pallets)
            hoists = random.randint(min_hoists, max_hoists)
            crates = random.randint(min_crates, max_crates)
            seed = random.randint(1, DepotsDomainGenerator.MAX_SEED)

            properties = {

                "min_depots": min_depots,
                "max_depots": max_depots,

                "min_distributors": min_distributors,
                "max_distributors": max_distributors,

                "min_trucks": min_trucks,
                "max_trucks": max_trucks,

                "min_pallets": min_pallets,
                "max_pallets": max_pallets,

                "min_hoists": min_hoists,
                "max_hoists": max_hoists,

                "min_crates": min_crates,
                "max_crates": max_crates,

                "depots": depots,
                "distributors": distributors,
                "trucks": trucks,
                "pallets": pallets,
                "hoists": hoists,
                "crates": crates,
                "seed": seed,

                "bin_params": ["depots", "trucks"]
            }

            file.write_properties(file_handle, properties,
                                  constants.PDDL_COMMENT_PREFIX)

            gen_cmd = "%s -e %u -i %u -t %u -p %u -h %u -c %u -s %u" % (
                DepotsDomainGenerator.GENERATOR_BIN,
                depots,
                distributors,
                trucks,
                pallets,
                hoists,
                crates,
                seed)

            unused_completed_process = subprocess.run(
                gen_cmd, shell=True, stdout=file_handle)

        # Just return an empty list.
        return []
