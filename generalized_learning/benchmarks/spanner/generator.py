
import pathlib
import random
import shutil
import subprocess

from benchmarks.generator import Generator
from util import constants
from util import file


class SpannerDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["min_spanners", "max_spanners",
                         "min_nuts", "max_nuts", "min_locations",
                         "max_locations"]) \
        .union(Generator.REQUIRED_KEYS)

    DOMAIN_FILE = (pathlib.Path(__file__).absolute().parent /
                   "domain.pddl").resolve()

    GENERATOR_BIN = (pathlib.Path(
        __file__).absolute().parent / "spanner.py").resolve()

    MIN_SPANNERS = 1
    MIN_NUTS = 1
    MIN_LOCATIONS = 1

    _DOMAIN_NAME = "spanner"

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(SpannerDomainGenerator, self).__init__(parent, parent_dir,
                                                     global_dict,
                                                     user_phase_dict,
                                                     failfast)

    def generate_domain(self):

        file_path = "%s/%s.domain.pddl" % (
            self._base_dir,
            SpannerDomainGenerator._DOMAIN_NAME)

        shutil.copy(SpannerDomainGenerator.DOMAIN_FILE,
                    file_path)

    def generate_problem(self, problem_range):

        min_spanners = self.get_value("min_spanners")
        max_spanners = self.get_value("max_spanners")
        min_nuts = self.get_value("min_nuts")
        max_nuts = self.get_value("max_nuts")
        min_locations = self.get_value("min_locations")
        max_locations = self.get_value("max_locations")

        assert min_spanners >= SpannerDomainGenerator.MIN_SPANNERS
        assert min_nuts >= SpannerDomainGenerator.MIN_NUTS
        assert min_locations >= SpannerDomainGenerator.MIN_LOCATIONS

        for problem_no in problem_range:

            problem_file = "problem_%u.problem.pddl" % (problem_no)
            file_handle = open("%s/%s" % (self._base_dir, problem_file), "w")

            spanners = random.randint(min_spanners, max_spanners)
            nuts = min(spanners, random.randint(min_nuts, max_nuts))
            locations = random.randint(min_locations, max_locations)

            properties = {

                "min_spanners": min_spanners,
                "max_spanners": max_spanners,
                "min_nuts": min_nuts,
                "max_nuts": max_nuts,
                "min_locations": min_locations,
                "max_locations": max_locations,
                "spanners": spanners,
                "nuts": nuts,
                "locations": locations,

                "bin_params": ["spanners", "nuts", "locations"]
            }

            file.write_properties(file_handle, properties,
                                  constants.PDDL_COMMENT_PREFIX)

            gen_cmd = "%s %u %u %u" % (
                SpannerDomainGenerator.GENERATOR_BIN,
                spanners,
                nuts,
                locations)

            unused_completed_process = subprocess.run(
                gen_cmd, shell=True, stdout=file_handle)

        # Just return an empty list.
        return []
