import copy
import pathlib
import sys
import random

sys.path.append("%s/../" % (pathlib.Path(__file__).parent))
import config

from pddlgym.core import PDDLEnv
from agent import PRPAgent
from model import Model
from utils import learning_utils
from interrogation.saia import AgentInterrogation
from utils.file_utils import FileUtils
from evaluators.result_logger import DiffResultsLogger
from planner import laostar
import tqdm
import os
import shutil
from model import UnconformantPreconditionException
from model import UnconformantEffectException
from exploration import drift_aware_bfs
from exploration import drift_explore
import pickle
from pddlgym.core import SimulatorOutOfBudgetException
from planner.prp import PRPPolicyNotFoundException

from evaluators.qace_evaluator import QaceEvaluator

DEBUG = False

class QaceStatelessEvaluator(QaceEvaluator):

    NAME = "qace-stateless"

    def __init__(self, base_dir,
                 sampling_count=5,
                 explore_mode="random_walk",
                 num_simulations=50,
                 num_rw_tries=25,
                 debug=True,
                 enable_time_thread=False,
                 failure_threshold=10):

        super(QaceStatelessEvaluator, self).__init__(
            base_dir, sampling_count=sampling_count,
            explore_mode=explore_mode,
            num_simulations=num_simulations,
            num_rw_tries=num_rw_tries,
            debug=debug,
            enable_time_thread=enable_time_thread,
            failure_threshold=failure_threshold)

    def get_name(self):

        return QaceStatelessEvaluator.NAME

    def solve_task(self, task_no, domain_file, problem_file, horizon=40,
        simulator_budget=5000):

        self.model = None
        super().solve_task(task_no, domain_file, problem_file,
                           horizon=horizon,
                           simulator_budget=simulator_budget)


if __name__ == "__main__":

    pass