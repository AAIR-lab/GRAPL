
import itertools
import logging
import math
import multiprocessing
import shutil

from abstraction.domain import AbstractDomain
from concretized.problem import Problem
from concretized.solution import Solution
from neural_net.nn import NN
from search.solver import Solver
from util import constants
from util import executor
from util import file
from util.iterator import flatten
from util.phase import Phase

from .state import AbstractState


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Model(Phase):

    #: The set of required keys for this phase in order to function correctly.
    REQUIRED_KEYS = set(["nn_type", "nn_name", "solver_name",
                         "input_dir"]) \
        .union(Phase.REQUIRED_KEYS)

    #: The default phase dict for this phase.
    DEFAULT_PHASE_DICT = {

        **Phase.DEFAULT_PHASE_DICT,

        "use_mpi": False,
        "max_workers": multiprocessing.cpu_count(),
        "chunk_size": 25,
        "force_single_core": False,
        "epochs": 200,
        "batch_size": 32,
        "shuffle": True,
        "train_retries": 5,
    }

    @staticmethod
    def get_instance(parent, parent_dir, global_dict, user_phase_dict,
                     failfast):

        return Model(parent, parent_dir, global_dict, user_phase_dict,
                     failfast)

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(Model, self).__init__(parent, parent_dir, global_dict,
                                    user_phase_dict, failfast)

    def train(self, model, abstract_domain, nn_train_pkgs_list):

        pass

    def _gen_get_training_data_args(self, chunk_size, domain_filepath,
                                    problem_file_list, solution_source,
                                    abstract_domain):

        assert chunk_size > 0

        total_problems = len(problem_file_list)
        assert total_problems > 0

        total_chunks = math.ceil(total_problems / chunk_size)
        logger.debug("Generating total_chunks=%u" % (total_chunks))

        for chunk_no in range(total_chunks):

            start = chunk_no * chunk_size
            end = min(total_problems, start + chunk_size)

            yield (domain_filepath, problem_file_list[start:end],
                   solution_source, abstract_domain)

    def _get_training_data(self, domain_filepath, problem_file_list,
                           solver_name, abstract_domain):

        if abstract_domain is None:

            abstract_domain = AbstractDomain()

        nn_pkgs_list = []
        for problem_filepath in problem_file_list:

            solution_filepath = "%s.%s.%s" % (problem_filepath,
                                              solver_name,
                                              constants.SOLUTION_FILE_EXT)

            problem = Problem(domain_filepath.name, problem_filepath.name,
                              problem_filepath.parent)
            if not problem._is_relaxed_reachable:

                continue

            try:
                solution = Solution.parse(solution_filepath)
            except Exception:

                continue

            current_state = problem.get_initial_state()

            action_list = solution.get_action_list()
            plan_length = len(action_list)
            for i in range(plan_length):

                abstract_state = AbstractState(problem, current_state)

                action_name = action_list[i]
                action = problem.get_action(action_name)
                nn_pkg = abstract_domain.encode_nn_training_data(
                    abstract_state,
                    action,
                    plan_length - i)

                nn_pkgs_list.append(nn_pkg)

                assert action.is_applicable(current_state)
                current_state = action.apply(current_state)

        return [(abstract_domain, nn_pkgs_list)]

    def get_training_data(self, problem_dir, max_workers, chunk_size, use_mpi,
                          force_single_core, abstract_domain=None):

        solver_name = self.get_value("solver_name")

        problem_list = file.get_file_list(problem_dir,
                                          constants.PROBLEM_FILE_REGEX)

        domain_list = file.get_file_list(problem_dir,
                                         constants.DOMAIN_FILE_REGEX)

        assert len(domain_list) == 1
        domain_filepath = domain_list[0]

        if force_single_core:

            training_data = executor.singlecore_execute(
                self._get_training_data,
                (domain_filepath, problem_list, solver_name, abstract_domain))
        else:

            training_data = executor.multicore_execute(
                self._get_training_data,
                (domain_filepath, problem_list, solver_name, abstract_domain),
                self._gen_get_training_data_args,
                max_workers, chunk_size,
                use_mpi)

        return training_data

    def _gen_remap_training_data_args(self, chunk_size,
                                      remapped_abstract_domain,
                                      training_data_list):

        assert chunk_size > 0

        total_problems = len(training_data_list)
        assert total_problems > 0

        total_chunks = math.ceil(total_problems / chunk_size)
        logger.debug("Generating total_chunks=%u" % (total_chunks))

        for chunk_no in range(total_chunks):

            start = chunk_no * chunk_size
            end = min(total_problems, start + chunk_size)

            yield (remapped_abstract_domain, training_data_list[start:end], )

    def _remap_training_data(self, remapped_abstract_domain,
                             training_data_list):

        nn_train_pkgs_list = []
        for training_data in training_data_list:

            old_abstract_domain = training_data[0]
            old_nn_train_pkgs_list = training_data[1]

            nn_train_pkgs_list += \
                remapped_abstract_domain.fix_nn_training_data(
                    old_abstract_domain, old_nn_train_pkgs_list)

        return nn_train_pkgs_list

    def remap_training_data(self, remapped_abstract_domain, training_data_list,
                            max_workers, chunk_size, use_mpi,
                            force_single_core):

        if force_single_core:

            remapped_training_data = executor.singlecore_execute(
                self._remap_training_data,
                (remapped_abstract_domain, training_data_list))
        else:

            remapped_training_data = executor.multicore_execute(
                self._remap_training_data,
                (remapped_abstract_domain, training_data_list),
                self._gen_remap_training_data_args,
                max_workers, chunk_size,
                use_mpi)

        return remapped_training_data

    def _execute(self):

        max_workers = self.get_value("max_workers")
        chunk_size = self.get_value("chunk_size")
        use_mpi = self.get_value("use_mpi")

        force_single_core = self.get_value("force_single_core")

        self.initialize_directories()
        try:
            shutil.rmtree("/tmp/train")
            shutil.rmtree("/tmp/eval")
            shutil.rmtree("/tmp/predict")
        except:
            pass
        # Get the training data.
        training_dir = file.get_relative_path(self.get_value("input_dir"),
                                              self._parent_dir)

        training_data = self.get_training_data(training_dir, max_workers,
                                               chunk_size, use_mpi,
                                               force_single_core, None)

        # Form the complete abstract domain.
        abstract_domains_iter = itertools.starmap(
            lambda abstract_domain, _: abstract_domain, training_data)

        abstract_domain = AbstractDomain.merge_abstract_domains(
            abstract_domains_iter)
        abstract_domain.initialize_nn_parameters()

        # Remap the training data.
        remapped_training_data = self.remap_training_data(abstract_domain,
                                                          training_data,
                                                          max_workers,
                                                          chunk_size, use_mpi,
                                                          force_single_core)

#         for i in range(len(remapped_training_data)):
#
#             print(i, "******************************************************")
#             for layer in remapped_training_data[i]._layers:
#
#                 print("===== LAYER =====", layer)
#                 print(remapped_training_data[i].decode(layer))

        # Train the network.
        train_try = 0
        should_train = True
        while should_train and train_try < self.get_value("train_retries"):

            train_try += 1
            logger.info("Training network: Attempt %u/%u" % (
                train_try, self.get_value("train_retries")))

            nn = NN.get_instance(abstract_domain,
                                 self.get_value("nn_type"),
                                 self.get_value("nn_name"))
            nn.plot_model(self._base_dir)
            should_train = not nn.train(remapped_training_data,
                                        self.get_value("epochs"),
                                        self.get_value("batch_size"),
                                        self.get_value("shuffle"))[0]

        # Evaluate the network.
        if self.has_key("evaluate_dir"):

            test_dir = file.get_relative_path(self.get_value("evaluate_dir"),
                                              self._parent_dir)

            test_data = self.get_training_data(test_dir, max_workers,
                                               chunk_size, use_mpi,
                                               force_single_core,
                                               abstract_domain)

            test_pkgs_list = []
            test_pkgs_list += itertools.starmap(
                lambda _, test_pkgs_list: test_pkgs_list, test_data)
            test_pkgs_list = list(flatten(test_pkgs_list))

            nn.evaluate(test_pkgs_list)

        # Save the model.
        nn.save(self._base_dir)

        return []

    def execute(self):

        # https://stackoverflow.com/questions/42504669/keras-tensorflow-and-multiprocessing-in-python
        # It seems that keras does not play well with multiprocessing fork().
        #
        # Thus, we will do all keras operations from a subprocess of main().
        executor.multicore_execute(
            self._execute,
            (),
            lambda chunk_size: (tuple(), ),
            max_workers=1,
            chunk_size=float("nan"),
            use_mpi=False)

        return []

# Import all classes needed for get_instance() here.
# We can't import it at the top since that would make cyclic imports.
