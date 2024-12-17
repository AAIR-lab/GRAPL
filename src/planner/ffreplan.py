'''
Created on Jan 14, 2023

@author: rkaria
'''


import copy
import pathlib
import sys
sys.path.append("%s/../" % (pathlib.Path(__file__).parent))

import config
import tempfile
import os

from model import Model
import subprocess
import re

from pddlgym import parser as pddlgym_parser
from planner.ff import FF

class FFReplan:
    
    EXECUTABLE = "%s/dependencies/FF-v2.3modified/ff" % (
        config.PROJECT_ROOTDIR)
    
    SOLN_FILE_EXTENSION = "soln"
    LOG_FILE_EXTENSION = "log"
    ACTION_SEPARATOR = " "
    
    @staticmethod
    def get_reward(execution_trace):
        
        if len(execution_trace) == 0:
            
            return 0
        else:
            
            return execution_trace[-1][3]
    
    def __init__(self, gym_domain_name, model=None, output_dir=None,
                 store_logs=True):
        
        self.ff = FF(gym_domain_name, 0, model=model)
        
        if store_logs:
            
            assert output_dir is not None
        
        self.output_dir = output_dir
        self.logs_dir = self.get_log_dir(0)
    
    def get_log_dir(self, problem_idx):
        
        return "%s/ffreplan_logs_%u" % (self.output_dir, problem_idx)
    
    def get_total_problems(self):
        
        return self.ff.get_total_problems()
    
    def set_problem_idx(self, problem_idx):
        
        self.ff.set_problem_idx(problem_idx)
        self.logs_dir = self.get_log_dir(problem_idx)
        
    def get_domain_filename(self):
        
        return "ff_replan_domain.pddl"
    
    def get_problem_filename(self, step):
        
        return "problem_step%s.pddl" % (step)
    
    def run_replanning(self, max_steps=sys.maxsize,
                       step=1):
        
        domain_filename = self.get_domain_filename()
        problem_filename = self.get_problem_filename(step)
        
        initial_state = None
        done = False
        
        execution_trace = []
        for _ in range(max_steps):
            
            transitions = self.ff.simulate_in_gym(
                output_dir=self.logs_dir,
                domain_filename=domain_filename,
                problem_filename=problem_filename,
                total_steps=step,
                initial_state=initial_state)
            
            execution_trace += transitions
            
            if len(transitions) == 0 or transitions[-1][4]:
                
                break
            
            initial_state = transitions[-1][2]
            
        return execution_trace
 
    
    @staticmethod
    def simple_example(domain_name):

        # FileUtils.initialize_directory(config.RESULT_DIR)
        print("Storing results in %s" % (config.RESULT_DIR))

        ffreplan = FFReplan(domain_name, output_dir=config.RESULT_DIR)
        execution_trace = ffreplan.run_replanning(max_steps=80)
        reward = 0.0 if len(execution_trace) == 0 else execution_trace[-1][3]
        
        print(reward)


if __name__ == "__main__":
    
    from utils import FileUtils
    import pddlgym
    import gym
    import utils
    from model import Model

    FFReplan.simple_example("Tireworld")