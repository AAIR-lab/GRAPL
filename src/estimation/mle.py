'''
Created on Jan 6, 2023

@author: rkaria
'''


import pathlib
import sys
sys.path.append("%s/../" % (pathlib.Path(__file__).parent))

import config
import copy
from pddlgym.structs import Literal
from pddlgym.structs import LiteralConjunction
from pddlgym.structs import ProbabilisticEffect

from enum import Enum

class MLE:
    
    EFFECTS = Enum("EFFECTS", ["NOP_OR_UNKNOWN", "AMBIGUOUS_OR_NOP",
                               "UNKNOWN_NOP_EFFECT",
                               "UNKNOWN_EFFECT"])
    
    @staticmethod
    def is_action_in_std_format(action):
        
        action_is_std = True
        action_is_std &= isinstance(action.effects, ProbabilisticEffect)
        
        assert action.effects.is_flattened
        assert action.effects.is_optimized
        
        for literal_conjunction in action.effects.literals:
            
            action_is_std &= isinstance(literal_conjunction, 
                                        LiteralConjunction)
            
            for literal in literal_conjunction.literals:
                
                action_is_std &= isinstance(literal, Literal)
        
        return action_is_std
            

    
    @staticmethod
    def label_episode_data(action_labels, model, episode,
                           obj_map,
                           inverse_naming_map={}):
        
        assert isinstance(episode, list)
        
        # The format of the episode data is [s, a, s', ..., s'']
        for i in range(1, len(episode) - 1, 2):
            
            print("========")
            # The format of an action is (name, [args[0], ... args[n]))
            assert isinstance(episode[i], tuple)
            pass
        
        
            action_args = PRP.get_pddlgym_args_from_prp_args(episode[i][1],
                                                             obj_map)
            
            action = model.actions[episode[i][0]]
            grounded_action = action.ground(action_args)
            
            if not action.effects.common_effects.holds(episode[i + 1].literals):
                
                print("UNKNOWN")
            
            for effect_no in range(len(grounded_action.effects.literals)):
                
                effect = grounded_action.effects.literals[effect_no]
                
                if not effect.holds(episode[i + 1].literals):
                    print("Unknown")
                
                if effect.holds(episode[i + 1].literals):
                    
                    print("YES", episode[i][0], episode[i][1], effect_no)
                else:
                    print("NO", episode[i][0], episode[i][1], effect_no)
        
        pass
    

if __name__ == "__main__":
    
    print("MLE example")

    import config
    import pddlgym
    import gym
    import utils
    from model import Model
    from utils import FileUtils
    from utils import helpers
    from planner.prp import PRP
    
    experiment = config.EXPERIMENTS[0]
    assert experiment["name"] == "tireworld"
    
    env = gym.make("PDDLEnv{}-v0".format(experiment["gym_domain_name"]))
    env.fix_problem_index(experiment["problem_idx"])
    _ = env.reset()
    
    domain, problem = utils.extract_elements(env, experiment["problem_idx"])
    
    model = Model(domain, clean=True)
    
    domain_file = "%s/domain.pddl" % (experiment["base_dir"])
    problem_file = "%s/problem.pddl" % (experiment["base_dir"])
    
    FileUtils.initialize_directory(experiment["base_dir"])
    
    model = model.flatten(with_copy=True)
    model = model.optimize(with_copy=False)
    
    model.write(domain_file)
    problem.write(problem_file, fast_downward_order=True)
    
    policy = PRP.solve(domain_file, problem_file)
    assert policy.is_goal_reachable()
    
    policy.transform_to_pddlgym(problem)
    
    episode = PRP.generate_pddlgym_samples_using_policy(
        env, domain, problem, policy,
        H=experiment["H"],
        naming_map=experiment["naming_map"],
        args_func_map=experiment["args_func_map"])
    
    # MLE stuff starts now!
    action_labels = {}
    obj_map = PRP.get_pddlgym_object_map(problem)
    inverse_naming_map = helpers.reverse_dict_with_hashable_values(
        experiment["naming_map"])

    # Sanity check for the action format expected in the model
    # data structure.
    for action in model.actions.values():
        
        assert MLE.is_action_in_std_format(action)

    MLE.label_episode_data(action_labels, model, episode,
                           obj_map,
                           inverse_naming_map=inverse_naming_map)
    