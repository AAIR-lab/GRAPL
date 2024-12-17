'''
Created on Jan 9, 2023

@author: rkaria
'''

import copy
import pathlib
import sys
sys.path.append("%s/../" % (pathlib.Path(__file__).parent))

import config
import argparse
import gym

from utils import FileUtils
import pddlgym
import gym
import utils
from model import Model
import os
import time
import csv
from evaluators.result_logger import ResultLogger


from glib.ndr import ndrs

from planner.laostar import LAOStar
import pickle

class GroundTruthEvaluator:
        
    def __init__(self, gym_domain_name):
        
        self.gym_domain_name = gym_domain_name

        self.env = gym.make("PDDLEnv{}Test-v0".format(gym_domain_name))
        self.domain = self.env.domain

        self.naming_map = {}
        self.args_func_map = {}
        self.ground_truth_model = Model(self.domain, clean=True)
        self.ground_truth_model = self.ground_truth_model.flatten(
            with_copy=False)
        self.ground_truth_model = self.ground_truth_model.optimize(
            with_copy=False)
    
    def compute_laostar_success_rate_and_cost(self, output_dir, 
                                              max_steps=50, 
                                              num_simulations=10):
        
        laostar = LAOStar(self.gym_domain_name,
                            model=self.ground_truth_model)
        
        total_cost = [0] * laostar.get_total_problems()
        total_reward = [0] * laostar.get_total_problems()
        
        agg_cost = 0
        agg_reward = 0
        
        for i in range(laostar.get_total_problems()):
            
            laostar.set_problem_idx(i)
            
            try:
                policy = laostar.solve(output_dir=output_dir)
            except Exception:
                
                policy = None
        
            for _ in range(num_simulations):
                
                if policy is not None:
                    transitions = laostar.gym_execute(policy,
                                                      total_steps=max_steps)
                else:
                    transitions = []

                if len(transitions) == 0 or transitions[-1][3] != 1.0:
                    
                    total_cost[i] += max_steps
                else: 
                    
                    total_reward[i] += 1 if transitions[-1][3] == 1.0 else 0
                    total_cost[i] += len(transitions)
                    
            agg_cost += total_cost[i]
            agg_reward += total_reward[i]
            
            total_cost[i] /= num_simulations
            total_reward[i] /= num_simulations
        
        agg_cost /= (laostar.get_total_problems() * num_simulations)
        agg_reward /= (laostar.get_total_problems() * num_simulations)
        return total_cost, total_reward, agg_cost, agg_reward
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog="Glib Runner",
        description="Run Glib and collect results")
    parser.add_argument("--results-dir", required=True, type=str,
                        help="The path to the results csv file")
    parser.add_argument("--gym-domain-name", required=True, type=str,
                        help="The gym domain name")
    
    args = parser.parse_args()
    
    gt_evalulator = GroundTruthEvaluator(args.gym_domain_name)
    print(gt_evalulator.compute_laostar_success_rate_and_cost(args.results_dir))

