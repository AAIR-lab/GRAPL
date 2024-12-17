'''
Created on Jan 9, 2023

@author: rkaria
'''

import copy
import pathlib
import sys
from dependencies.glib.planning_modules.ffreplan import FFReplanner
sys.path.append("%s/../" % (pathlib.Path(__file__).parent))

import config

import glib.main as glib
import argparse
from pddlgym.structs import ProbabilisticEffect
import gym

from utils import FileUtils
from utils import learning_utils
import pddlgym
import gym
import utils
from model import Model
import os
import time
import csv
import pickle

from evaluators.result_logger import ResultLogger
from planner.prp import PRP
from pddlgym import structs
from pddlgym.core import _select_operator as pddlgym_select_operator, PDDLEnv

from planner.ffreplan import FFReplan
from planner.laostar import LAOStar
from interrogation import AgentInterrogation
from agent import PRPAgent
from cafeworld_agent import CafeworldAgent

class SAIAEvaluator:
    

    def __init__(self, domain_file, problem_file,
                 result_logger,
                 vd_transitions=None,
                 disable_evaluator=False):

        self.domain_file = domain_file
        self.problem_file = problem_file

        self.result_logger = result_logger
        self.experiment_params = {
            "horizon": config.DEFAULT_HORIZON,
            "naming_map": {},
            "args_func_map": {},
        }

        self.vd_transitions = vd_transitions
        self.disable_evaluator = disable_evaluator
        assert self.vd_transitions is not None or self.disable_evaluator

        self.env = PDDLEnv(self.domain_file, self.problem_file,
                           operators_as_actions=True,
            grounded_actions_in_state=config.SHOULD_FIND_GYM_ACTION_PREDS)
        self.domain = self.env.domain
        
        self.ground_truth_model = Model(self.domain)
        self.ground_truth_model = self.ground_truth_model.flatten(
            with_copy=False)
        self.ground_truth_model = self.ground_truth_model.optimize(
            with_copy=False)
        
        self.action_map = PRP.get_pddlgym_action_map(self.domain)
        self.best_vd = float("inf")
     

    def get_effect_idx_corresponding_to_transition(self, s, a, s_dash, action):
    
        effect_idx = None
        for idx, expected_s_dash in enumerate(action.effects.apply_all(s)):
            
            if expected_s_dash == s_dash:

                if effect_idx is not None:
                    
                    print("Action: ", action.name)
                    print("Action Params: ", action.params)
                    print("Optimization", action.is_optimized)
                    print("Precon", action.preconds)
                    print("Common Effects", action.effects.common_effects)
                    print("Effects", action.effects)
                    assert effect_idx is None

                effect_idx = idx
        
        return effect_idx
    
    def get_ground_truth_probability_range_for_transition(self, s, a, s_dash):

        operator, assignment = pddlgym_select_operator(s, a, self.domain)

        # If the action is not applicable, then we will reach s_dash with
        # a probability of 1.0.
        if operator is None:
            
            assert s.literals == s_dash.literals
            return 1.0
            

        args = []
        for param in operator.params:
            
            args.append(assignment[param])

        action = self.ground_truth_model.actions[a.predicate.name]
        action = action.ground(args, with_copy=True)
        
        effect_idx = self.get_effect_idx_corresponding_to_transition(
            s.literals, a, s_dash.literals, action)
        assert effect_idx is not None

        p = action.effects.get_probability(effect_idx)
        return p

    def _new_vd_samples(self, evaluation_model):

        vd = 0
        for s, a, s_dash, execution_status in self.vd_transitions:

            # First, get the ground-truth operator.
            operator, assignment = pddlgym_select_operator(s, a, self.domain)

            # Add both sets of action predicates that can be received from
            # pddlgym to the states. This removes any intricacies associated
            # with pddlgym.
            #
            # We do this to prevent the s.literals == s_dash.literals
            # errorneously failing due to weird pddlgym stuff.
            s = s.literals | {a}
            s_dash = s_dash.literals | {a}

            # Next, ground the action we have in the transition
            # per the model and check if the preconditions hold.
            action = evaluation_model.actions[a.predicate.name]
            action = action.ground(a.variables, with_copy=True)

            precondition_holds = action.preconds.holds(s)
            # No operator found in the ground truth,
            # VD = 0 iff the precondition does not hold.
            if not execution_status:

                assert operator is None
                assert s == s_dash
                vd += precondition_holds
            else:

                # Precondition holds

                # Ensure the vd transition generator
                # returned a success status here.
                assert execution_status

                # Sample an effect.
                expected_s_dash = action.effects.apply(s)

                # VD = 0 only if the precondition holds
                # and the expected s_dash matches that
                # in the transition.
                vd += not (precondition_holds and expected_s_dash == s_dash)

        return vd / len(self.vd_transitions)

    def check_transition(self, s, a, s_dash, evaluation_model,
                         is_ground_truth=False):

        operator, assignment = pddlgym_select_operator(s, a, self.domain)

        # If the action is not applicable, then we will reach s_dash with
        # a probability of 1.0.
        if operator is None:
            
            assert s.literals == s_dash.literals
            return (0, 1.0)
            

        # Add both sets of action predicates that can be received from
        # pddlgym to the states. This removes any intricacies associated 
        # with pddlgym.
        s = set(s.literals) | {a} 
        s_dash = s_dash.literals  | {a}

        args = []
        for param in operator.params:
            
            args.append(assignment[param])

        action = evaluation_model.actions[a.predicate.name]
        action = action.ground(args, with_copy=True)
        precondition_holds = action.preconds.holds(s)
        
        if precondition_holds:

            # Test for ground truth model to make sure that one effect
            # holds if all preconditions hold.
            
            if is_ground_truth:
                
                assert self.get_effect_idx_corresponding_to_transition(s, a,
                    s_dash, action) is not None
            
            assert isinstance(action.effects, ProbabilisticEffect)
            expected_s_dash = action.effects.apply(s)
            effect_idx = self.get_effect_idx_corresponding_to_transition(
                s, a, s_dash, action)
            p = action.effects.get_probability(effect_idx) \
                    if effect_idx is not None else 0.0
            
            vd = 1.0 if expected_s_dash != s_dash else 0.0 
            return (vd, p) 
        else:
            
            # Only enable if testing ground truth.
            assert not is_ground_truth or s == s_dash
            vd, p = (0.0, 1.0) if s == s_dash else (1.0, 0)
            vd, p = (0.0, 0.0)
        
            return (vd, p)
        pass
    
    def compute_variational_distance(self, evaluation_model, is_ground_truth):
        
        vd_sample = 0
        vd_p = 0
        for s, a, s_dash in self.vd_transitions:
            
            vd, p = self.check_transition(s, a, s_dash, evaluation_model,
                                        is_ground_truth)
            
            gt_p = self.get_ground_truth_probability_range_for_transition(
                s, a, s_dash)
            
            vd_sample += vd
            vd_p += abs(gt_p - p)
            
        return vd_sample / len(self.vd_transitions), \
            vd_p / len(self.vd_transitions)
    
    def compute_ffreplan_success_rate(self, output_dir, evaluation_model,
                             max_steps=50):
        
        ffreplan = FFReplan(self.gym_domain_name, output_dir=output_dir,
                            model=evaluation_model)
        
        total_reward = 0
        for i in range(ffreplan.get_total_problems()):
            
            ffreplan.set_problem_idx(i)
            execution_trace = ffreplan.run_replanning(max_steps=max_steps,
                                                      step=1)
            
            total_reward += FFReplan.get_reward(execution_trace)
            
        return total_reward / ffreplan.get_total_problems()
    
    def compute_laostar_success_rate_and_cost(self, output_dir, 
                                              evaluation_model,
                                              max_steps=50, 
                                              num_simulations=10):
        
        laostar = LAOStar(self.gym_domain_name,
                            model=evaluation_model)
        
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
                    
                    try:
                        transitions = laostar.gym_execute(policy,
                                                          total_steps=max_steps)
                    except Exception:
                        
                        transitions = []
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
    
    def saia_callback(self, **kwargs):
        
        if self.disable_evaluator:
            
            return
        
        query_number = kwargs["query_number"]
        evaluation_model = copy.deepcopy(kwargs["evaluation_model"])
        evaluation_model = evaluation_model.flatten(with_copy=False)
        evaluation_model = evaluation_model.optimize(with_copy=False)
        
        is_ground_truth = kwargs.get("is_ground_truth", False)
        
        domain_filepath = "%s/valid_domain_model.pddl" % (kwargs["output_dir"])
        evaluation_model.write(domain_filepath, with_probabilities=True)
        
        elapsed_time = kwargs["elapsed_time"]
        
        itr = kwargs["itr"]
        num_pal_tuples = kwargs["num_pal_tuples"]
        output_dir = kwargs["output_dir"]

        vd, vd_p = self.compute_variational_distance(evaluation_model,
                                                    is_ground_truth)

        if vd < self.best_vd:

            self.best_vd = vd

        vd_p = None
        success_rate = None
        total_cost = None
        total_reward = None
        agg_cost = None
        agg_reward = None

        self.enable_ffreplan = False
        if self.enable_ffreplan:
            success_rate = self.compute_ffreplan_success_rate(output_dir,
                                                              evaluation_model)

        self.enable_lao = False
        if self.enable_lao:
            total_cost, total_reward, agg_cost, agg_reward = \
                self.compute_laostar_success_rate_and_cost(output_dir,
                                                           evaluation_model)

        self.result_logger.log_results((num_pal_tuples, query_number), 
                                        itr, success_rate, vd,
                                       self.best_vd, elapsed_time,
                                       total_cost, 
                                       total_reward, 
                                       agg_cost, 
                                       agg_reward)

    def run(self, args, **kwargs):
        
        data_dir = kwargs["data_dir"]

        if self.domain.domain_name.lower() == "cafeworld":
            agent = CafeworldAgent(args.gym_domain_name,
                                   big_problem=False,
                                   params=self.experiment_params,
                                   base_dir=data_dir)
        else:
            env = PDDLEnv(args.domain_file, args.problem_file,
                          operators_as_actions=True,
                          shared_instance=True)
            env.reset()
            agent = PRPAgent(env,
                             self.experiment_params)

        domain = agent.get_domain()
        problem = agent.get_problem()

        agent_model = Model(domain, clean=True)
        abstract_actions = learning_utils.abstract_action(domain.operators)
        abstract_model = Model(domain, actions=abstract_actions)
        
        query_files_dir = data_dir
        interrogation = AgentInterrogation(agent, abstract_model, problem,
                                           query_files_dir, agent_model,\
                                           self.saia_callback,
                                           args.sampling_count, 
                                           randomize=args.randomize_pal,
                                           should_count_sdm_samples=args.count_sdm_samples,
                                           explore_mode=args.explore_mode)
        learned_model = interrogation.interrogate()
        
        model = learned_model.flatten(with_copy=True)
        model = model.optimize(with_copy=False)
        model.predicates = agent_model.predicates
        model.write(data_dir + "/domain.pddl", 
                    with_probabilities=True)
    
if __name__ == "__main__":
    
    experiment = {
        "name": "tireworld",
        "gym_domain_name": "Explodingblocks",
        "problem_idx" : 2,
        "base_dir": "%s/tireworld" % (config.RESULT_DIR),
        "H": 30,
        "naming_map": {},
        "args_func_map": {}
    }
    
    saia_evaluator = SAIAEvaluator(experiment["base_dir"],
                                   experiment["gym_domain_name"],
                                   experiment["naming_map"],
                                   experiment["args_func_map"])
    
    from model import Model
    ground_truth_model = Model(saia_evaluator.domain)
    
    saia_evaluator.saia_callback(evaluation_model=ground_truth_model,
                                 itr=0,
                                 num_pal_tuples=None,
                                 output_dir=config.RESULT_DIR,
                                 elapsed_time=0,
                                 is_ground_truth=True)
