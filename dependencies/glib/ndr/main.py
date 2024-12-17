"""Full data gathering, learning and planning pipeline
"""
from envs.ndr_blocks import NDRBlocksEnv, noiseoutcome
from envs.pybullet_blocks import PybulletBlocksEnv
from learn import run_main_search
from planning import find_policy
from utils import run_policy, get_env_id
from pddlgym.structs import Anti
import gym
import pddlgym
from collections import defaultdict
from termcolor import colored
import pickle
import os
import numpy as np


def collect_training_data(env, outfile=None, verbose=False, **kwargs):
    """Load or generate training data
    """
    if outfile is not None and os.path.exists(outfile):
        with open(outfile, 'rb') as f:
            transition_dataset = pickle.load(f)
        num_transitions = sum(len(v) for v in transition_dataset.values())
        print("Loaded {} transitions for {} actions.".format(num_transitions, 
            len(transition_dataset)))
    else:
        print("Collecting transition data... ", end='')
        transition_dataset = collect_transition_dataset(env, verbose=verbose, **kwargs)
        num_transitions = sum(len(v) for v in transition_dataset.values())
        print("collected {} transitions for {} actions.".format(num_transitions, 
            len(transition_dataset)))
        if outfile is not None:
            with open(outfile, 'wb') as f:
                pickle.dump(transition_dataset, f)
            print("Dumped dataset to {}.".format(outfile))
    return transition_dataset

def collect_transition_dataset(env, max_num_trials=5000, num_transitions_per_problem=1,
                               max_transitions_per_action=500, policy=None, actions="all", 
                               verbose=False):
    """Collect transitions (state, action, effect) for the given actions
    Make sure that no more than 50% of outcomes per action are null.
    """
    if actions == "all":
        actions = env.action_predicates

    total_counts = { a : 0 for a in actions }
    num_no_effects = { a : 0 for a in actions }

    if policy is None:
        policy = lambda s : env.action_space.sample(s)
    transitions = defaultdict(list)
    for trial in range(max_num_trials):
        if all(c >= max_transitions_per_action for c in total_counts.values()):
            break
        if verbose:
            print("\nCollecting data trial {}".format(trial))
        done = True
        for transition_num in range(num_transitions_per_problem):
            if verbose:
                # print("Transition {}/{}".format(transition_num, num_transitions_per_problem))
                print("total counts:", total_counts)
                print("num_no_effects:", num_no_effects)
            if done:
                obs, _ = env.reset()
            action = policy(obs)
            next_obs, _, done, _ = env.step(action)
            effects = construct_effects(obs, next_obs)

            null_effect = len(effects) == 0 or noiseoutcome() in effects
            keep_transition = (actions == "all" or action.predicate in actions) and \
                (not null_effect or (num_no_effects[action.predicate] < \
                    total_counts[action.predicate]/2.+1)) and \
                 total_counts[action.predicate] < max_transitions_per_action
            if keep_transition:
                total_counts[action.predicate] += 1
                if null_effect:
                    num_no_effects[action.predicate] += 1
                transition = (obs.literals, action, effects)
                transitions[action.predicate].append(transition)
            obs = next_obs

    return transitions

def construct_effects(obs, next_obs):
    """Convert a next observation into effects
    """
    # This is just for debugging environments where noise outcomes are simulated
    if noiseoutcome() in next_obs.literals:
        return { noiseoutcome() }
    effects = set()
    for lit in next_obs.literals - obs.literals:
        effects.add(lit)
    for lit in obs.literals - next_obs.literals:
        effects.add(Anti(lit))
    return effects

def print_transition(transition):
    print("  State:", transition[0])
    print("  Action:", transition[1])
    print("  Effects:", transition[2])

def print_training_data(training_data):
    for action, transitions_for_action in training_data.items():
        print(colored(action, attrs=['bold']))
        for transition in transitions_for_action:
            print_transition(transition)
            print()

def learn_rule_set(training_data, outfile=None, search_method="greedy", verbose=False):
    """Main learning step
    """
    if outfile is not None and os.path.exists(outfile):
        with open(outfile, 'rb') as f:
            rules = pickle.load(f)
        num_rules = sum(len(v) for v in rules.values())
        if verbose:
            print("Loaded {} rules for {} actions.".format(num_rules, len(rules)))
    else:
        if verbose:
            print("Learning rules... ")
        rules = run_main_search(training_data, search_method=search_method)
        num_rules = sum(len(v) for v in rules.values())
        if verbose:
            print("Learned {} rules for {} actions.".format(num_rules, len(rules)))
        if outfile is not None:
            with open(outfile, 'wb') as f:
                pickle.dump(rules, f)
            if verbose:
                print("Dumped rules to {}.".format(outfile))
    if verbose:
        print_rule_set(rules)
    return rules

def print_rule_set(rule_set):
    for action_predicate in sorted(rule_set):
        print(colored(action_predicate, attrs=['bold']))
        for rule in rule_set[action_predicate]:
            print(rule)

def run_test_suite(rule_set, env, outfile=None, num_problems=10, seed_start=10000, max_num_steps=50,
                   num_trials_per_problem=1, render=True, verbose=False, try_cache=False):
    if try_cache and os.path.exists(outfile):
        with open(outfile, 'rb') as f:
            all_returns = pickle.load(f)
    else:
        all_returns = []
        for seed in range(seed_start, seed_start+num_problems):
            seed_returns = []
            for trial in range(num_trials_per_problem):
                env.seed(seed)
                env.reset()
                policy = find_policy("ff_replan", rule_set, env.action_space, 
                    env.observation_space)
                total_returns = 0
                outdir = '/tmp'
                if render:
                    os.makedirs(outdir, exist_ok=True)
                returns = run_policy(env, policy, verbose=verbose, render=render, check_reward=False, 
                    max_num_steps=max_num_steps, outdir=outdir)
                seed_returns.append(returns)
            all_returns.append(seed_returns)
        if outfile is not None:
            with open(outfile, 'wb') as f:
                pickle.dump(all_returns, f)
            print("Dumped test results to {}.".format(outfile))
    print("Average returns:", np.mean(all_returns))
    return all_returns



def main():
    seed = 0

    training_env = PybulletBlocksEnv(use_gui=False)  #record_low_level_video=True, video_out='/tmp/lowlevel_training.mp4')
    # training_env = gym.make("PDDLEnvBlocks-v0")
    # training_env = gym.make("PDDLEnvHanoi-v0")
    # training_env = gym.make("PDDLEnvTsp-v0")
    # training_env = gym.make("PDDLEnvDoors-v0")
    # training_env = gym.make("PDDLEnvRearrangement-v0")
    # training_env = gym.make("PDDLEnvFerry-v0")
    # training_env.seed(seed)
    data_outfile = "data/{}_training_data.pkl".format(get_env_id(training_env))
    training_data = collect_training_data(training_env, data_outfile, verbose=True,
        max_num_trials=5000, #5000, 
        num_transitions_per_problem=10,
        max_transitions_per_action=500,)
    training_env.close()

    # print_training_data(training_data)

    rule_set_outfile = "data/{}_rule_set.pkl".format(get_env_id(training_env))
    rule_set = learn_rule_set(training_data, rule_set_outfile, search_method="greedy")

    # test_env = PybulletBlocksEnv(record_low_level_video=True, video_out='/tmp/lowlevel_test.gif') 
    # test_env = gym.make("PDDLEnvBlocksTest-v0")
    # test_env = gym.make("PDDLEnvHanoiTest-v0")
    # test_env = gym.make("PDDLEnvDoorsTest-v0")
    # test_env = gym.make("PDDLEnvTspTest-v0")
    # test_env = gym.make("PDDLEnvRearrangementTest-v0")
    # # test_env = gym.make("PDDLEnvFerryTest-v0")
    test_env = PybulletBlocksEnv(use_gui=False) 
    test_outfile = "data/{}_test_results.pkl".format(get_env_id(test_env))
    test_results = run_test_suite(rule_set, test_env, test_outfile, render=False, verbose=True,
        num_problems=1,
        max_num_steps=100)
    test_env.close()

    print("Test results:")
    print(test_results)


if __name__ == "__main__":
    main()
