'''
Created on Mar 26, 2020

@author: rkaria
'''

import datetime
import os
import pathlib
import re
import time

from matplotlib import rcParams

from util import git


def get_current_date():

    return str(datetime.date.today())


def get_current_time():

    current_time = time.localtime()
    return time.strftime("%H:%M:%S", current_time)


def set_experiment_name(name):

    global EXPERIMENT_NAME
    EXPERIMENT_NAME = name


def get_solution_file_regex(solver_name):

    global PROBLEM_FILE_EXT
    global SOLUTION_FILE_EXT

    if solver_name is None:

        solver_regex_str = "(\w|\W)*"
    else:

        solver_regex_str = ".(\w|\W)*%s(\w|\W)*" % (solver_name)

    return re.compile("(\w|\W)*.%s%s.%s($|\n)" % (
        PROBLEM_FILE_EXT,
        solver_regex_str,
        SOLUTION_FILE_EXT))


#: The root directory of the project.
ROOT_DIR = (pathlib.Path(__file__).parent / "../../").absolute()

HOSTNAME = os.uname().nodename

GIT_SHA = git.get_head_commit_sha(ROOT_DIR)
GIT_BRANCH = git.get_active_branch(ROOT_DIR)
GIT_IS_DIRTY = git.is_dirty(ROOT_DIR)

EXPERIMENT_NAME = "default"
EXPERIMENT_ID = time.strftime("%Y%m%d%H%M%S", time.localtime())

PROBLEM_FILE_EXT = "problem.pddl"
DOMAIN_FILE_EXT = "domain.pddl"
SOLUTION_FILE_EXT = "soln"
LOG_FILE_EXT = "log"

PROBLEM_FILE_REGEX = re.compile("(\w|\W)*.%s($|\n)" % (PROBLEM_FILE_EXT))
DOMAIN_FILE_REGEX = re.compile("(\w|\W)*.%s($|\n)" % (DOMAIN_FILE_EXT))
SOLUTION_FILE_REGEX = get_solution_file_regex(None)

PDDL_COMMENT_PREFIX = ";"

TAG_TYPES = True
TAG_BINARY_GOALS = True
TAG_UNARY_GOALS = True

USE_ARTIFICIAL_G = True

# Setup all plots to be in times new roman.
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times New Roman']

# Ignore the warnings from having many images open.
rcParams["figure.max_open_warning"] = 1000
