
import logging
import math
import multiprocessing

from tqdm.auto import trange

from generalized_learning.concretized.problem import Problem
from generalized_learning.plot import policy_results
from generalized_learning.qlearning.evaluator.policy_evaluator import PolicyEvaluator
from generalized_learning.qlearning.evaluator.random_evaluator import RandomPolicyEvaluator
from generalized_learning.qlearning.results import CSVResults
from util import constants
from util import executor
from util import file
from util.phase import Phase


logger = logging.getLogger(__name__)


class Evaluator(Phase):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["type", "input_dir"]).union(Phase.REQUIRED_KEYS)

    #: The default phase dict for this phase.
    DEFAULT_PHASE_DICT = {

        **Phase.DEFAULT_PHASE_DICT,

        "num_episodes": 1,
        "timesteps_per_episode": 500,
        "simulator_type": "generic",

        "use_mpi": False,
        "max_workers": multiprocessing.cpu_count(),
        "chunk_size": 25,
        "force_single_core": False
    }

    @staticmethod
    def get_instance(parent, parent_dir, global_dict, user_phase_dict, failfast):

        return Evaluator(parent, parent_dir, global_dict, user_phase_dict,
                         failfast)

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(Evaluator, self).__init__(parent, parent_dir, global_dict,
                                        user_phase_dict, failfast)

    def generate_args(self, chunk_size, domain_file, problem_list):

        assert chunk_size > 0

        total_problems = len(problem_list)
        assert total_problems > 0

        total_chunks = math.ceil(total_problems / chunk_size)
        if total_chunks < self.get_value("max_workers"):

            chunk_size = math.ceil(
                total_problems / self.get_value("max_workers"))
            total_chunks = math.ceil(total_problems / chunk_size)

        logger.debug("Generating total_chunks=%u" % (total_chunks))

        for chunk_no in range(total_chunks):

            start = chunk_no * chunk_size
            end = min(total_problems, start + chunk_size)

            yield (domain_file, problem_list[start:end], )

    def _get_evaluator(self):

        algorithm = self._phase_dict["type"]

        if "policy" == algorithm:

            assert "model_dir" in self._phase_dict
            return PolicyEvaluator(self._phase_dict, self._parent_dir)
        elif "random" == algorithm:

            return RandomPolicyEvaluator(self._phase_dict)
        else:

            raise Exception("Unknown evaluator method algorithm.")

    def evaluate(self, domain_file, problem_list):

        for problem_file in problem_list:

            evaluator = self._get_evaluator()
            problem = Problem(domain_file.name, problem_file.name,
                              problem_file.parent)

            evaluator.setup_globals(problem)
            evaluator.evaluate(problem)

        return []

    def single_core_evaluate(self, domain_file, problem_list):

        csv_results = CSVResults(
            problem_list[0].parent,
            fieldnames=["problem", "abstraction", "episode",
                        "solved", "cost"],
            output_prefix="policy_results")

        evaluator = self._get_evaluator()

        num_episodes = self._phase_dict["num_episodes"]
        timesteps_per_episode = self._phase_dict["timesteps_per_episode"]

        for problem_no in trange(len(problem_list),
                                 desc='Problem',
                                 unit="problem"):

            problem_file = problem_list[problem_no]

            problem = Problem(domain_file.name, problem_file.name,
                              problem_file.parent)

            for episode in trange(num_episodes,
                                  desc='Episode',
                                  unit="episode", leave=False):

                _, total_cost, done = evaluator.evaluate(
                    problem,
                    timesteps_per_episode)

                csv_results.add_data({
                    "problem": problem_file.name,
                    "abstraction":  self._phase_dict.get("nn_name", "random"),
                    "episode": episode,
                    "solved": done,
                    "cost": total_cost,
                })

        csv_results.close()
        policy_results.plot_policy_results(csv_results.output_filepath)

    def execute(self):

        #         max_workers = self.get_value("max_workers")
        #         chunk_size = self.get_value("chunk_size")
        #         use_mpi = self.get_value("use_mpi")
        #
        #         force_single_core = self.get_value("force_single_core")

        problem_dir = file.get_relative_path(self.get_value("input_dir"),
                                             self._parent_dir)

        problem_list = file.get_file_list(problem_dir,
                                          constants.PROBLEM_FILE_REGEX)

        domain_list = file.get_file_list(problem_dir,
                                         constants.DOMAIN_FILE_REGEX)

        assert len(domain_list) == 1
        domain_file = domain_list[0]

        self.single_core_evaluate(domain_file, problem_list)

        results = []
#         if force_single_core:
#
#             results = executor.singlecore_execute(
#                 self.evaluate, (domain_file, problem_list))
#         else:
#
#             results = executor.multicore_execute(
#                 self.evaluate,
#                 (domain_file, problem_list),
#                 self.generate_args,
#                 max_workers, chunk_size,
#                 use_mpi)

        return results


# Import all classes needed for get_instance() here.
# We can't import it at the top since that would make cyclic imports.
