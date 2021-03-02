
import pathlib
import random
import shutil
import subprocess
import sys

from benchmarks.generator import Generator
from util import constants
from util import file


class LogisticsDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["min_airplanes", "max_airplanes",
                         "min_cities", "max_cities",
                         "min_city_size", "max_city_size",
                         "min_packages", "max_packages"]) \
        .union(Generator.REQUIRED_KEYS)

    DOMAIN_FILE = (pathlib.Path(__file__).absolute().parent /
                   "domain.pddl").resolve()

    GENERATOR_BIN = (pathlib.Path(
        __file__).absolute().parent / "logistics").resolve()

    MIN_AIRPLANES = 1
    MIN_CITIES = 2
    MIN_CITY_SIZE = 1
    MIN_PACKAGES = 1
    MAX_SEED = (1 << sys.int_info.bits_per_digit) - 1

    _DOMAIN_NAME = "logistics"

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(LogisticsDomainGenerator, self).__init__(parent, parent_dir,
                                                       global_dict,
                                                       user_phase_dict,
                                                       failfast)

    def generate_domain(self):

        file_path = "%s/%s.domain.pddl" % (
            self._base_dir,
            LogisticsDomainGenerator._DOMAIN_NAME)

        shutil.copy(LogisticsDomainGenerator.DOMAIN_FILE,
                    file_path)

    def generate_problem(self, problem_range):

        min_airplanes = self.get_value("min_airplanes")
        max_airplanes = self.get_value("max_airplanes")
        min_cities = self.get_value("min_cities")
        max_cities = self.get_value("max_cities")
        min_city_size = self.get_value("min_city_size")
        max_city_size = self.get_value("max_city_size")
        min_packages = self.get_value("min_packages")
        max_packages = self.get_value("max_packages")

        assert min_airplanes >= LogisticsDomainGenerator.MIN_AIRPLANES
        assert min_cities >= LogisticsDomainGenerator.MIN_CITIES
        assert min_city_size >= LogisticsDomainGenerator.MIN_CITY_SIZE
        assert min_packages >= LogisticsDomainGenerator.MIN_PACKAGES

        for problem_no in problem_range:

            problem_file = "problem_%u.problem.pddl" % (problem_no)
            file_handle = open("%s/%s" % (self._base_dir, problem_file), "w")

            airplanes = random.randint(min_airplanes, max_airplanes)
            cities = random.randint(min_cities, max_cities)
            city_size = random.randint(min_city_size, max_city_size)
            packages = random.randint(min_packages, max_packages)
            seed = random.randint(1, LogisticsDomainGenerator.MAX_SEED)

            properties = {

                "min_airplanes": min_airplanes,
                "max_airplanes": max_airplanes,
                "min_cities": min_cities,
                "max_cities": max_cities,
                "min_city_size": min_city_size,
                "max_city_size": max_city_size,
                "min_packages": min_packages,
                "max_packages": max_packages,
                "airplanes": airplanes,
                "cities": cities,
                "city_size": city_size,
                "packages": packages,

                "bin_params": ["airplanes", "cities", "packages"]
            }

            file.write_properties(file_handle, properties,
                                  constants.PDDL_COMMENT_PREFIX)

            gen_cmd = "%s -a %u -c %u -s %u -p %u -r %u" % (
                LogisticsDomainGenerator.GENERATOR_BIN,
                airplanes,
                cities,
                city_size,
                packages,
                seed)

            unused_completed_process = subprocess.run(
                gen_cmd, shell=True, stdout=file_handle)

        # Just return an empty list.
        return []
