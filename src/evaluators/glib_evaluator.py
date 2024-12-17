'''
Created on Jan 9, 2023

@author: rkaria
'''

import copy
import pathlib
import sys
sys.path.append("%s/../" % (pathlib.Path(__file__).parent))

import config

import glib.main as glib
import argparse
from pddlgym.structs import ProbabilisticEffect
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

from glib.curiosity_modules.curiosity_base import BaseCuriosityModule
from pddlgym.core import _select_operator as pddlgym_select_operator
from pddlgym import parser as pddlgym_parser
from pddlgym.structs import LiteralConjunction

from glib.ndr import ndrs

from planner.laostar import LAOStar
import pickle

class GLIBEvaluator:
    
    NAME = "glib"
    
    def __init__(self, gym_domain_name, seed, curiosity_name, learning_name,
                 result_logger, output_dir, vd_transitions, 
                 disable_evaluation=False):
        
        self.gym_domain_name = gym_domain_name
        self.seed = seed
        self.curiosity_name = curiosity_name
        self.learning_name = learning_name
        self.result_logger = result_logger
        self.output_dir = output_dir
        self.vd_transitions = vd_transitions
        self.disable_evaluation = disable_evaluation
        
        self.metadata = None

        self.glib_runner = glib.get_glib_runner_from_saia(
            self.seed,
            self.gym_domain_name,
            self.curiosity_name,
            self.learning_name,
            self.glib_callback,
            self.vd_transitions)
        
        self.env = gym.make("PDDLEnv{}-v0".format(gym_domain_name))
        self.domain = self.env.domain

        self.naming_map = {}
        self.args_func_map = {}
        self.ground_truth_model = Model(self.domain)
        self.ground_truth_model = self.ground_truth_model.flatten(
            with_copy=False)
        self.ground_truth_model = self.ground_truth_model.optimize(
            with_copy=False)

        self.best_vd = float("inf")

    def create_operators_from_ndrset(self, ndrset):
        
        operators = []
        for i, ndr in enumerate(ndrset):
            
            if ndr is ndrset.default_ndr:
                
                continue
            
            op_name = "%s%s" % (ndrset.action.predicate.name, i)
            probabilities, effects = ndr.effect_probs, ndr.effects
            
            new_probs = []
            new_effects = []
            all_effects_for_params = []
            for i in range(len(probabilities)):
                
                if probabilities[i] > 0:
                    
                    assert ndrs.NOISE_OUTCOME not in effects[i]
                    new_probs.append(probabilities[i])
                    all_effects_for_params += effects[i]
                    new_effects.append(LiteralConjunction(effects[i]))
                    
            probabilistic_effect = ProbabilisticEffect(new_effects, new_probs)
            probabilistic_effect = probabilistic_effect.flatten(with_copy=False)
            
            # Do not add ndrset.action as a precondition.
            # Save it in the operator and we will restore it when running
            # laostar.
            preconds = LiteralConjunction(sorted(ndr.preconditions))
            params = sorted({ v for lit in preconds.literals + all_effects_for_params \
                             for v in lit.variables })
            operator = pddlgym_parser.Operator(op_name, params, preconds,
                                                probabilistic_effect)
            operator.action_predicate = ndrset.action
            
            operator = operator.flatten(with_copy=False)
            operator = operator.optimize(with_copy=False)
            operators.append(operator)
        
        return operators

    def get_model_for_ndrs(self, ndrs):
        
        # The types, predicates etc remain the same.
        # Only operators are not visible and we override the ground truth's
        # operators with those from the ndr.
        ndr_model = copy.deepcopy(self.ground_truth_model)
        
        operators = []
        for ndrset in ndrs.values():
            
            operators += self.create_operators_from_ndrset(ndrset)
        
        ndr_model.actions = {operator.name : operator for operator in operators}
        
        ndr_model.is_glib = True
        return ndr_model

    def get_effect_idx_corresponding_to_transition(self, s, a, s_dash, action):
    
        effect_idx = None
        for idx, expected_s_dash in enumerate(action.effects.apply_all(s)):
            
            if expected_s_dash == s_dash:
                
                assert effect_idx is None
                effect_idx = idx
        
        return effect_idx
    
    def get_ground_truth_probability_range_for_transition(self, s, a, s_dash):

        operator, assignment = pddlgym_select_operator(s, a, self.domain)

        # If the action is not applicable, then we will reach s_dash with
        # a probability of 1.0.
        if operator is None:
            
            assert s.literals == s_dash.literals
            return (1.0, 1.0)
            

        args = []
        for param in operator.params:
            
            args.append(assignment[param])

        action = self.ground_truth_model.actions[a.predicate.name]
        action = action.ground(args, with_copy=True)
        
        effect_idx = self.get_effect_idx_corresponding_to_transition(
            s.literals, a, s_dash.literals, action)
        assert effect_idx is not None

        p = action.effects.get_probability(effect_idx)
        return (p, p)

    def test_saia_model_on_glib_transitions(self, model):
        
        self.glib_runner.learned_operators = set(model.operators)
        
    def run(self, args):
        
        self.glib_runner.run()
    
    def _get_ndr_probability(self, ndr, s, a, s_dash):

        min_p = float("inf")
        max_p = float("-inf")
        
        if ndr.covers_transition((s.literals, a, s_dash)):
            
            for i in range(len(ndr.effect_probs)):
                
                predicted_literals = ndr._predict(s.literals, a, i)
                predicted_s_dash = BaseCuriosityModule._execute_effects(
                    s, predicted_literals)
            
                if predicted_s_dash.literals == s_dash.literals:
                    
                    min_p = min(min_p, ndr.effect_probs[i])
                    max_p = max(max_p, ndr.effect_probs[i])

        return min_p, max_p

    def get_ndr_probability_range_for_action(self, s, a, s_dash, ndrs):

        min_p = float("inf")
        max_p = float("-inf")
        action_found = False
        
        for act_pred, ndrs in ndrs.items():
            if act_pred.name != a.predicate.name:
                continue
            
            action_found = True
            for ndr in ndrs:
                
                # Only update probabilities with the default ndr if
                # there have been no matches yet.
                #
                # The default NDR is always applicable and will keep s == s_dash
                # with probability 1.0
                #
                # The default NDR is always guaranteed to be the last ndr.
                if ndr is ndrs.default_ndr and min_p == float("inf"):
                    
                    _min_p, _max_p = (1.0, 1.0) if s.literals == s_dash.literals \
                        else (0, 0)
                    _min_p, _max_p = (0.0, 0.0)
                else:
                    _min_p, _max_p = self._get_ndr_probability(ndr, s, a, 
                                                               s_dash)
                
                min_p = min(min_p, _min_p)
                max_p = max(max_p, _max_p)
        
        # Use a default NDR if no ndr with the action has been found yet.
        if not action_found:
            
            # return (1.0, 1.0) if s.literals == s_dash.literals else (0, 0)
            return (0.0, 0.0)
        else:
            return (min_p, max_p)

    def _find_ndrset_for_action(self, ndrs, action):

        for action_predicate, ndrset in ndrs.items():

            if action_predicate.name == action.predicate.name:

                return ndrset

        return None

    def _get_rule(self, ndrset, s, a):

        rule = ndrset.find_rule((s, a, None))

        # If the default ndr is returned, then
        # we only return it if nothing else has
        # been learned.
        #
        # Optimism in the face of uncertainty.
        #
        # If there are other ndrs with no preconditions,
        # they will always be returned before the
        # default ndr.
        if rule == ndrset.default_ndr:
            for ndr in ndrset:
                if len(ndr.preconditions) > 0:
                    return None

        return rule

    def _new_vd_samples(self, ndrs):

        vd_success = [0, 0]
        vd_failure = [0, 0]
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

            ndrset = self._find_ndrset_for_action(ndrs, a)
            if ndrset is None:
                if execution_status:
                    vd_success[0] = vd_success[0] + 1
                    vd_success[1] = vd_success[1] + 1
                else:
                    vd_failure[0] = vd_failure[0] + 1
                    vd_failure[1] = vd_failure[1] + 1

                continue

            rule = self._get_rule(ndrset, s, a)
            precondition_holds = rule is not None

            # No operator found in the ground truth,
            # VD = 0 iff the precondition does not hold.
            if not execution_status:

                assert operator is None
                assert s == s_dash
                vd_failure[0] = vd_failure[0] + precondition_holds
                vd_failure[1] = vd_failure[1] + 1
            else:

                # Precondition holds

                # Ensure the vd transition generator
                # returned a success status here.
                assert operator is not None
                if not precondition_holds:

                    vd = 1.0
                else:
                    # Sample an effect.
                    effects = LiteralConjunction(rule.predict_sample(s, a))
                    expected_s_dash = effects.apply(s)

                    # VD = 0 only if the precondition holds
                    # and the expected s_dash matches that
                    # in the transition.
                    vd = not (expected_s_dash == s_dash)

                vd_success[0] = vd_success[0] + vd
                vd_success[1] = vd_success[1] + 1

        return vd_success[0] / vd_success[1], \
            vd_failure[0] / vd_failure[1]

    def compute_variational_distance(self, ndrs, 
                                     variational_distance_transitions):
        
        vd_min = 0
        vd_max = 0
        for s, a, s_dash in variational_distance_transitions:
            
            gt_min_p, gt_max_p = \
                self.get_ground_truth_probability_range_for_transition(s, a, 
                                                                       s_dash)
            
            min_p, max_p = self.get_ndr_probability_range_for_action(s, a, 
                                                                     s_dash,
                                                                     ndrs)
                
            vd_min += abs(gt_min_p - min_p)
            vd_max += abs(gt_max_p - max_p)
    
        vd_min = vd_min/ len(variational_distance_transitions)
        vd_max = vd_max / len(variational_distance_transitions)
        
        return (vd_min, vd_max)
    
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
    
    def glib_callback(self, **kwargs):
        
        if self.disable_evaluation:
            
            return
        
        ndrs = kwargs["ndrs"]
        itr = kwargs["itr"]
        success_rate = kwargs["success_rate"]
        variational_difference = kwargs["variational_distance"]
        elapsed_time = kwargs["elapsed_time"]

        if variational_difference < self.best_vd:

            self.best_vd = variational_difference

        vd_gt = None
        total_cost = None
        total_reward = None
        agg_cost = None
        agg_reward = None

        ndr_model = self.get_model_for_ndrs(ndrs)
        model_dir = "%s/itr_%u/" % (self.output_dir, itr)
        # FileUtils.initialize_directory(model_dir, clean=False)

        # ndr_model.write("%s/domain.pddl" % (model_dir), with_probabilities=True)

        self.enable_gt = False
        if self.enable_gt:
            vd_transitions = self.glib_runner.get_variational_dist_transitions()
        
            vd_gt = self.compute_variational_distance(
                ndrs, vd_transitions)

        self.enable_lao = False
        if self.enable_lao:
            total_cost, total_reward, agg_cost, agg_reward = \
                self.compute_laostar_success_rate_and_cost(
                    "%s/itr_%u" % (self.output_dir, itr),
                    ndr_model)
            
        # with open("%s/itr_%u/ndrs.pkl" % (self.output_dir, itr), "wb") as f:
        #     pickle.dump(ndrs, f)

        self.result_logger.log_results(self.metadata, itr, success_rate,
                                       variational_difference, 
                                       self.best_vd,
                                       elapsed_time,
                                       total_cost, 
                                       total_reward, 
                                       agg_cost, 
                                       agg_reward)
        
    def update_glib_model(self, model):
        
        learned_operators = set([action for action in model.actions.values()])
        self.glib_runner.agent.learned_operators = learned_operators
            
        self.glib_runner.agent._planning_module._ff_planner._learned_operators = \
            learned_operators
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog="Glib Runner",
        description="Run Glib and collect results")
    parser.add_argument("--results-dir", required=True, type=str,
                        help="The path to the results csv file")
    parser.add_argument("--clean", default=False, action="store_true",
                        help="Whether to clean the results file or not.")
    parser.add_argument("--experiment-name", default=None, type=str,
                        help="The name of the experiment")
    parser.add_argument("--gym-domain-name", required=True, type=str,
                        help="The gym domain name")
    parser.add_argument("--curiosity-name", required=True, type=str,
                        choices=["GLIB_G1", "GLIB_L2"],
                        help="The name of the curiosity routine for glib")
    parser.add_argument("--learning-name", default="LNDR", type=str,
                        choices=["LNDR"],
                        help="The name of the learning routine for glib")
    parser.add_argument("--generate-only",
                        action="store_true",
                        default=False,
                        help="Only generate the vd transitions")
    
    parser.add_argument("--seed", default=None, type=int,
                        help="The seed to use")
    
    args = parser.parse_args()
    
    if args.experiment_name is None:
        
        args.experiment_name = "glib_%s_%s_%s" % (args.gym_domain_name,
                                                    args.curiosity_name,
                                                    args.learning_name)
    
    args.experiment_name = args.experiment_name.lower()
    
    result_logger = ResultLogger(args.results_dir, 
                                 filename=args.experiment_name,
                                 clean_file=args.clean)
    
    if args.seed is None:
        
        args.seed = int(time.time())
    
    glib_evaluator = GLIBEvaluator(args.gym_domain_name, 
                                   args.seed,
                                   args.curiosity_name,
                                   args.learning_name,
                                   result_logger,
                                   args.results_dir)

    if not args.generate_only:
        glib_evaluator.run_glib()
