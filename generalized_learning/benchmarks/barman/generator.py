
import pathlib
import random
import shutil
import subprocess

from benchmarks.generator import Generator
from util import constants
from util import file


class BarmanDomainGenerator(Generator):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["domain_name", "min_cocktails", "max_cocktails",
                         "min_ingredients", "max_ingredients",
                         "min_shots", "max_shots"]) \
        .union(Generator.REQUIRED_KEYS)

    DOMAIN_FILE = (pathlib.Path(__file__).absolute().parent /
                   "domain.pddl").resolve()

    GENERATOR_BIN = (pathlib.Path(
        __file__).absolute().parent / "barman.py").resolve()

    MIN_COCKTAILS = 1
    MIN_INGREDIENTS = 2
    MIN_SHOTS = 1

    _DOMAIN_NAME = "barman"

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(BarmanDomainGenerator, self).__init__(parent, parent_dir,
                                                    global_dict,
                                                    user_phase_dict,
                                                    failfast)

    def generate_domain(self):

        file_path = "%s/%s.domain.pddl" % (
            self._base_dir,
            BarmanDomainGenerator._DOMAIN_NAME)

        shutil.copy(BarmanDomainGenerator.DOMAIN_FILE,
                    file_path)

    def generate_problem(self, problem_range):

        min_cocktails = self.get_value("min_cocktails")
        max_cocktails = self.get_value("max_cocktails")
        min_ingredients = self.get_value("min_ingredients")
        max_ingredients = self.get_value("max_ingredients")
        min_shots = self.get_value("min_shots")
        max_shots = self.get_value("max_shots")

        assert min_cocktails >= BarmanDomainGenerator.MIN_COCKTAILS
        assert min_ingredients >= BarmanDomainGenerator.MIN_INGREDIENTS
        assert min_shots >= BarmanDomainGenerator.MIN_SHOTS

        for problem_no in problem_range:

            problem_file = "problem_%u.problem.pddl" % (problem_no)
            file_handle = open("%s/%s" % (self._base_dir, problem_file), "w")

            cocktails = random.randint(min_cocktails, max_cocktails)
            ingredients = random.randint(min_ingredients, max_ingredients)
            shots = random.randint(min_shots, max_shots)

            properties = {

                "min_cocktails": min_cocktails,
                "max_cocktails": max_cocktails,
                "min_ingredients": min_ingredients,
                "max_ingredients": max_ingredients,
                "min_shots": min_shots,
                "max_shots": max_shots,

                "cocktails": cocktails,
                "ingredients": ingredients,
                "shots": shots,
            }

            file.write_properties(file_handle, properties,
                                  constants.PDDL_COMMENT_PREFIX)

            gen_cmd = "%s %u %u %u" % (
                BarmanDomainGenerator.GENERATOR_BIN,
                cocktails,
                ingredients,
                shots)

            unused_completed_process = subprocess.run(
                gen_cmd, shell=True, stdout=file_handle)

        # Just return an empty list.
        return []
