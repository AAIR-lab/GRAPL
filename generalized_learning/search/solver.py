
import logging
import math
import multiprocessing

from search.astar import AStar
from search.informed import InformedSearch
from util import constants
from util import executor
from util import file
from util.phase import Phase

from .fd import FD
from .ff import FF
from .pyperplan import Pyperplan

logger = logging.getLogger(__name__)


class Solver(Phase):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["type", "input_dir"]).union(Phase.REQUIRED_KEYS)

    #: The default phase dict for this phase.
    DEFAULT_PHASE_DICT = {

        **Phase.DEFAULT_PHASE_DICT,

        "use_mpi": False,
        "max_workers": multiprocessing.cpu_count(),
        "chunk_size": 25,
        "force_single_core": False
    }

    @staticmethod
    def get_instance(parent, parent_dir, global_dict, user_phase_dict, failfast):

        return Solver(parent, parent_dir, global_dict, user_phase_dict,
                      failfast)

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(Solver, self).__init__(parent, parent_dir, global_dict,
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

    def _get_search_algorithm(self):

        algorithm = self._phase_dict["type"]
        name = self._phase_dict["name"]

        if "ff" == algorithm:

            return FF(name, self._phase_dict)
        elif "fd" == algorithm:

            return FD(name, self._phase_dict)
        elif "pyperplan" == algorithm:

            return Pyperplan(name, self._phase_dict, self._parent_dir)
        elif "informed" == algorithm:

            return InformedSearch(name, self._phase_dict,
                                  self._parent_dir)
        elif "astar" == algorithm:

            return AStar(name, self._phase_dict,
                         self._parent_dir)
        else:

            raise Exception("Unknown search algorithm.")

    def solve(self, domain_file, problem_list):

        search_algorithm = self._get_search_algorithm()

        solutions = []
        for problem_file in problem_list:

            search_algorithm._reset()
            solution = search_algorithm.search(domain_file, problem_file)
            solutions.append(solution)

        return solutions

    def execute(self):

        max_workers = self.get_value("max_workers")
        chunk_size = self.get_value("chunk_size")
        use_mpi = self.get_value("use_mpi")

        force_single_core = self.get_value("force_single_core")

        problem_dir = file.get_relative_path(self.get_value("input_dir"),
                                             self._parent_dir)

        problem_list = file.get_file_list(problem_dir,
                                          constants.PROBLEM_FILE_REGEX)

        domain_list = file.get_file_list(problem_dir,
                                         constants.DOMAIN_FILE_REGEX)

        assert len(domain_list) == 1
        domain_file = domain_list[0]

        if force_single_core:

            results = executor.singlecore_execute(
                self.solve, (domain_file, problem_list))
        else:

            results = executor.multicore_execute(
                self.solve,
                (domain_file, problem_list),
                self.generate_args,
                max_workers, chunk_size,
                use_mpi)

        return results


# Import all classes needed for get_instance() here.
# We can't import it at the top since that would make cyclic imports.
