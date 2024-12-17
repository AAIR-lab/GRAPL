import pathlib
import sys
from enum import Enum, IntEnum

class Location(IntEnum):
    PRECOND = 1
    EFFECTS = 2
    ALL = 3

class Literal(Enum):
    AN = -2
    NEG = -1
    ABS = 0
    POS = 1
    AP = 2
    NP = 3

PLANNER = "PRP"
PROJECT_ROOTDIR = pathlib.Path("%s/../" % (pathlib.Path(__file__).parent)).as_posix()

GLIB_ROOTDIR = pathlib.Path("%s/%s" % (PROJECT_ROOTDIR, "dependencies/glib/")).as_posix()

# Add paths so that glib and glib specific files can be directly used from our code base.
sys.path.append(pathlib.Path("%s/%s" % (PROJECT_ROOTDIR, "dependencies/")).as_posix())
sys.path.append(GLIB_ROOTDIR)

# Add the path to pddlgym-0.0.5 so that we use our own custom installation of
# pddlgym. Furthermore, append it to the front so that we bypass any system 
# installation of pddlgym.
sys.path.insert(0, pathlib.Path("%s/%s" % (PROJECT_ROOTDIR, "dependencies/pddlgym-0.0.5")).as_posix())

RESULT_DIR = "%s/results" % (PROJECT_ROOTDIR)

# Configure the benchmarks dir.
BENCHMARKS_DIR = "%s/benchmarks" % (PROJECT_ROOTDIR)

ITRS_PER_PRECOND = 1
SAMPLING_COUNT = 5

domains = ["exploding-blocksworld"]
transition_count = 100

SHOULD_FIND_GYM_ACTION_PREDS = False
OVERRIDE_ACTION_EFFECTS_WHEN_ANALYZING_SAMPLES = True
SDMA_MODE = False

DEFAULT_HORIZON = 40

NAMING_MAP = {
    
    "Tireworld": {},
    "Tireworld_og": {},
    "Explodingblocks": {},
    "Explodingblocks_og": {},
    "Probabilistic_elevators": {},
    "First_responders": {},
    "Cafeworld": {},
}

ARGS_FUNC_MAP = {
    
    "Tireworld": {},
    "Tireworld_og": {"move-car": lambda x: [x[1]]},
    "Explodingblocks": {},
    "Explodingblocks_og": {"pick-up": lambda x: [x[0]],
                          "put-down": lambda x: [x[0]],
                          "stack": lambda x: [x[0], x[1]],
                          "unstack": lambda x: [x[0]]},
    "Probabilistic_elevators": {},
    "First_responders": {},
    "Cafeworld": {},
}

EXPERIMENTS = [
    
    {
        "name": "tireworld",
        "gym_domain_name": "Tireworld",
        "problem_idx" : 2,
        "base_dir": "%s/tireworld" % (RESULT_DIR),
        "H": 30,
        "naming_map": {},
        "args_func_map": {}
    },
    {
        "name": "tireworld_og",
        "gym_domain_name": "Tireworld_og",
        "problem_idx" : 2,
        "base_dir": "%s/tireworld_og" % (RESULT_DIR),
        "H": 30,
        "naming_map": {},
        "args_func_map": {"move-car": lambda x: [x[1]]}
    },
    {
        "name": "exploding-blocksworld",
        "gym_domain_name": "Explodingblocks",
        "problem_idx": 1,
        "base_dir": "%s/explodingblocks" % (RESULT_DIR),
        "H": 30,
        "naming_map": {},
        "args_func_map": {}
    },
    {
        "name": "exploding-blocksworld",
        "gym_domain_name": "Explodingblocks_og",
        "problem_idx": 1,
        "base_dir": "%s/explodingblocks_og" % (RESULT_DIR),
        "H": 30,
        "naming_map": {},
        "args_func_map": {"pick-up": lambda x: [x[0]],
                          "put-down": lambda x: [x[0]],
                          "stack": lambda x: [x[0], x[1]],
                          "unstack": lambda x: [x[0]]}
    },
    {
        "name": "probabilistic_elevators",
        "gym_domain_name": "Probabilistic_elevators",
        "problem_idx" : 0,
        "base_dir": "%s/probabilistic_elevators" % (RESULT_DIR),
        "H": 30,
        "naming_map": {},
        "args_func_map": {}
    },
    {
        "name": "first_responders",
        "gym_domain_name": "First_responders",
        "problem_idx" : 0,
        "base_dir": "%s/first_responders" % (RESULT_DIR),
        "H": 30,
        "naming_map": {},
        "args_func_map": {}
    },
]