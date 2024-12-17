'''
Created on Jan 13, 2023

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

import pddlgym
import gym
import utils
from utils.file_utils import FileUtils

class FF:
    
    EXECUTABLE = "%s/dependencies/FF-v2.3modified/ff" % (
        config.PROJECT_ROOTDIR)
    
    SOLN_FILE_EXTENSION = "soln"
    LOG_FILE_EXTENSION = "log"
    ACTION_SEPARATOR = " "
    
    def __init__(self, gym_domain_name, problem_idx, model=None):
        
        self.env = gym.make(
            "PDDLEnv{}Test-v0".format(gym_domain_name),
            grounded_actions_in_state=config.SHOULD_FIND_GYM_ACTION_PREDS)
        
        self.problem_idx = problem_idx
        
        self.env.fix_problem_index(problem_idx)
        _ = self.env.reset()
        
        self.domain, self.problem = utils.extract_elements(self.env, 
                                                 self.problem_idx)
        
        if model is None:
            self.model = Model(self.domain)
        else:
            
            self.model = copy.deepcopy(model)
            
        self.model = self.model.flatten(with_copy=False)
        self.model = self.model.optimize(with_copy=False)
        self.model = self.model.determinize(with_copy=False)
        
    
    def get_total_problems(self):
        
        return len(self.env.problems)
    
    def set_problem_idx(self, problem_idx):
        
        self.problem_idx = problem_idx
        self.env.fix_problem_index(problem_idx)
        _ = self.env.reset()
        
        _, self.problem = utils.extract_elements(self.env, self.problem_idx)
    
    def sanitize(self, output_dir, domain_filename, problem_filename):
        
        if not output_dir:
            
            output_dir = tempfile.TemporaryDirectory()
            
        if not os.path.exists(output_dir):
            FileUtils.initialize_directory(output_dir)
            
        assert os.path.isdir(output_dir)

        if not domain_filename:
            
            domain_filename = "ff_domain.pddl"
            
        if not problem_filename:
            
            problem_filename = "ff_problem.pddl"
            
        assert os.path.split(domain_filename)[0] == ""
        assert os.path.split(problem_filename)[0] == ""
        
        return output_dir, domain_filename, problem_filename
    
    def write_files(self, output_dir, domain_filename, problem_filename,
                    initial_state_literals):
        
        domain_filepath = "%s/%s" % (output_dir, domain_filename) 
        problem_filepath = "%s/%s" % (output_dir, problem_filename)
        solution_filepath = "%s/%s.%s" % (output_dir, problem_filename,
                                          FF.SOLN_FILE_EXTENSION)
        log_filepath = "%s/%s.%s" % (output_dir, problem_filename,
                                     FF.LOG_FILE_EXTENSION)
        
        if os.path.exists(solution_filepath):
            
            os.remove(solution_filepath)
        
        self.model.write(domain_filepath)
        self.problem.write(problem_filepath,
                           initial_state=initial_state_literals,
                           fast_downward_order=True)
        
        ff_cmd = "%s -o %s -f %s" % (FF.EXECUTABLE, domain_filepath, 
                                     problem_filepath)
        
        return ff_cmd, solution_filepath, log_filepath

    def solve(self, output_dir=None, 
              domain_filename=None,
              problem_filename=None,
              timelimit_in_sec=60,
              initial_state_literals=None,
              raise_exception=False):
        
        output_dir, domain_filename, problem_filename = \
            self.sanitize(output_dir, domain_filename, problem_filename)
            
        cmd_string, solution_filepath, log_filepath = self.write_files(
            output_dir,  domain_filename,  problem_filename, 
            initial_state_literals)
        
        stdout_filehandle = open(log_filepath, "w")
        
        try:
            
            subprocess.run(cmd_string, shell=True, check=True,
                           cwd=output_dir,
                           stdout=stdout_filehandle,
                           stderr=subprocess.STDOUT,
                           timeout=timelimit_in_sec)
        except Exception as e:
            
            if raise_exception:
                
                raise(e)
            
            return []
        
        return self.parse_plan(solution_filepath)
    
    def parse_plan(self, solution_filepath):

        if not os.path.exists(solution_filepath):
            
            return []

        solution_file = open(solution_filepath, "r")
        first = True
        plan = []
        for line in solution_file:
            
            if first:
                first = False
                continue

            line = line.strip()
            line = line.lower()
            line = line.replace("(", "")
            line = line.replace(")", "")

            plan.append(line)

        return plan
    
    def simulate_in_gym(self, output_dir=None,
                        domain_filename=None,
                        problem_filename=None,
                        timelimit_in_sec=None,
                        initial_state=None,
                        total_steps=float("inf")):
        
        if initial_state is not None:
            initial_state_literals = initial_state.literals
        else:
            initial_state_literals = None

        plan = self.solve(output_dir=output_dir,
                          domain_filename=domain_filename,
                          problem_filename=problem_filename,
                          timelimit_in_sec=timelimit_in_sec,
                          initial_state_literals=initial_state_literals)
        
        return self.gym_execute(plan, total_steps=total_steps,
                                initial_state=initial_state)
    
    def gym_execute(self, plan, total_steps=float("inf"),
                    initial_state=None):

        _ = self.env.reset()
        if initial_state is not None:
            
            self.env.set_state(initial_state)
            
        s = self.env.get_state()
        transitions = []
        
        done = False
        for i in range(min(total_steps, len(plan))):
            
            a = pddlgym_parser.parse_ff_plan_step(plan[i], self.domain, 
                                                       self.problem)
            
            if a is None:
                
                break
            
            s_dash, r, done, _ = self.env.step(a)
        
            transitions.append((s, a, s_dash, r, done))
            
            if done:
                
                break
            
            s = s_dash
            
        return transitions
    
    @staticmethod
    def simple_example(domain_name, problem_index):

        # FileUtils.initialize_directory(config.RESULT_DIR)
        print("Storing results in %s" % (config.RESULT_DIR))

        ff = FF(domain_name, problem_index)
        plan = ff.solve(output_dir=config.RESULT_DIR)
        
        print(plan)
    
    @staticmethod
    def simple_execution_in_pddlgym(domain_name, problem_index,
                                    episode_timesteps=80):
        
        # FileUtils.initialize_directory(config.RESULT_DIR)
        print("Storing results in %s" % (config.RESULT_DIR))
     
        ff = FF(domain_name, problem_index)
        transitions = ff.simulate_in_gym(output_dir=config.RESULT_DIR)
        
        print(transitions)
        print(transitions[-1][3])
            

if __name__ == "__main__":
    
    from utils import FileUtils
    import pddlgym
    import gym
    import utils
    from model import Model

    FF.simple_execution_in_pddlgym("Tireworld", 0)