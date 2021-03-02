
from abc import abstractmethod
import copy
import logging
import math
import multiprocessing

from util import executor
from util.phase import Phase

logger = logging.getLogger(__name__)


class Generator(Phase):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "total_problems" to be specified.
    REQUIRED_KEYS = set(["type", "total_problems"]).union(Phase.REQUIRED_KEYS)

    #: The default phase dict for this phase.
    DEFAULT_PHASE_DICT = {

        **Phase.DEFAULT_PHASE_DICT,

        "use_mpi": False,
        "max_workers": multiprocessing.cpu_count(),
        "chunk_size": 25,
        "force_single_core": False
    }

    @staticmethod
    def get_instance(parent, parent_dir, global_dict, user_phase_dict,
                     failfast):

        generator_type = user_phase_dict["type"]
        if "delivery" == generator_type:

            generator_cls = DeliveryDomainGenerator
        elif "miconic" == generator_type:

            generator_cls = MiconicDomainGenerator
        elif "visitall" == generator_type:

            generator_cls = VisitAllDomainGenerator
        elif "goldminer" == generator_type:

            generator_cls = GoldminerDomainGenerator
        elif "spanner" == generator_type:

            generator_cls = SpannerDomainGenerator
        elif "childsnack" == generator_type:

            generator_cls = ChildsnackDomainGenerator
        elif "ferry" == generator_type:

            generator_cls = FerryDomainGenerator
        elif "logistics" == generator_type:

            generator_cls = LogisticsDomainGenerator
        elif "grippers" == generator_type:

            generator_cls = GrippersDomainGenerator
        elif "gripper" == generator_type:

            generator_cls = GripperDomainGenerator
        elif "hanoi" == generator_type:

            generator_cls = HanoiDomainGenerator
        elif "grid" == generator_type:

            #             raise Exception("Current generator yields unsolvable instances.")
            generator_cls = GridDomainGenerator
        elif "npuzzle" == generator_type:

            generator_cls = NPuzzleDomainGenerator
        elif "blocksworld" == generator_type:

            generator_cls = BlocksworldDomainGenerator
        elif "barman" == generator_type:

            generator_cls = BarmanDomainGenerator
        elif "parking" == generator_type:

            generator_cls = ParkingDomainGenerator
        elif "tyreworld" == generator_type:

            generator_cls = TyreworldDomainGenerator
        elif "depots" == generator_type:

            generator_cls = DepotsDomainGenerator
        elif "sokoban" == generator_type:

            generator_cls = SokobanDomainGenerator
        elif "sokoban2" == generator_type:

            generator_cls = Sokoban2DomainGenerator
        else:

            raise Exception("Unknown generator type.")

        return generator_cls(parent, parent_dir, global_dict, user_phase_dict,
                             failfast)

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(Generator, self).__init__(parent, parent_dir, global_dict,
                                        user_phase_dict, failfast)

    def generate_args(self, chunk_size, problem_start_no):

        assert chunk_size > 0

        total_problems = self._phase_dict["total_problems"]
        assert total_problems > 0

        total_chunks = math.ceil(total_problems / chunk_size)
        logger.debug("Generating total_chunks=%u" % (total_chunks))

        for chunk_no in range(total_chunks):

            start = problem_start_no + chunk_no * chunk_size
            end = min(total_problems + problem_start_no, start + chunk_size)

            # (range(x, y)) expands to just range(x, y)
            # Use (range(x, y), ) to force it to remain (range(x, y), )
            yield (range(start, end), )

    @abstractmethod
    def generate_domain(self):

        raise NotImplementedError

    @abstractmethod
    def generate_problem(self, problem_range):

        raise NotImplementedError

    def execute(self):

        max_workers = self.get_value("max_workers")
        chunk_size = self.get_value("chunk_size")
        use_mpi = self.get_value("use_mpi")

        force_single_core = self.get_value("force_single_core")

        self.initialize_directories()

        # Generate the domain as well.
        self.generate_domain()

        # Minimum one iteration is always executed by the generator.
        max_index = 1

        # Detect how many iterations and which keys need to be replaced.
        iter_set = set()
        for key in self._phase_dict.keys():

            if isinstance(self._phase_dict[key], list):

                max_index = max(max_index, len(self._phase_dict[key]))
                iter_set.add(key)

        results = []

        # Get a copy of the phase dict with the lists.
        # These will be replaced on the fly.
        phase_dict_copy = copy.deepcopy(self._phase_dict)
        problem_start_no = 0
        for i in range(max_index):

            for key in iter_set:

                try:

                    self._phase_dict[key] = phase_dict_copy[key][i]
                except IndexError:

                    self._phase_dict[key] = phase_dict_copy[key][-1]

            if force_single_core:

                results += executor.singlecore_execute(
                    self.generate_problem, self.generate_args(float("inf"),
                                                              problem_start_no))
            else:

                results += executor.multicore_execute(self.generate_problem,
                                                      (problem_start_no, ),
                                                      self.generate_args,
                                                      max_workers, chunk_size,
                                                      use_mpi)

            # Increment the start problem no. for the next set.
            problem_start_no = problem_start_no \
                + phase_dict_copy["total_problems"]

        return results


# Import all classes needed for get_instance() here.
# We can't import it at the top since that would make cyclic imports.
from .barman.generator import BarmanDomainGenerator
from .blocksworld.generator import BlocksworldDomainGenerator
from .childsnack.generator import ChildsnackDomainGenerator
from .delivery.generator import DeliveryDomainGenerator
from .depots.generator import DepotsDomainGenerator
from .ferry.generator import FerryDomainGenerator
from .goldminer.generator import GoldminerDomainGenerator
from .grid.generator import GridDomainGenerator
from .gripper.generator import GripperDomainGenerator
from .grippers.generator import GrippersDomainGenerator
from .hanoi.generator import HanoiDomainGenerator
from .logistics.generator import LogisticsDomainGenerator
from .miconic.generator import MiconicDomainGenerator
from .npuzzle.generator import NPuzzleDomainGenerator
from .parking.generator import ParkingDomainGenerator
from .sokoban.generator import SokobanDomainGenerator
from .sokoban2.generator import Sokoban2DomainGenerator
from .spanner.generator import SpannerDomainGenerator
from .tyreworld.generator import TyreworldDomainGenerator
from .visitall.generator import VisitAllDomainGenerator
