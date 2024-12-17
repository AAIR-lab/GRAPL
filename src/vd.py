'''
Created on Jan 17, 2023

@author: rkaria
'''
import pathlib
import sys

from pddlgym.core import PDDLEnv

sys.path.append("%s/../" % (pathlib.Path(__file__).parent))
import config

import pickle
import random
import pddlgym
import gym

from agent import BFSAgent
import tqdm
import argparse
import os
import glob

class VariationalDistance:
    
    FILTER_FUNC = lambda s, a, s_dash, depth : a is not None
    STORAGE_FUNC = lambda s, a, s_dash, depth : (s, a, s_dash)
    
    @staticmethod
    def load_transitions(output_filepath):

        try:
            file_handle = open(output_filepath, "rb")
            transitions = pickle.load(file_handle)
            return transitions
        except Exception:

            return None

    @staticmethod
    def generate_samples(domain_file,
                         problem_file,
                         successful_actions, unsuccessful_actions,
                         max_visited_states=4000,
                         pbar_lvl=0):

        env = PDDLEnv(domain_file, problem_file,
                      operators_as_actions=True,
                      raise_error_on_invalid_action=False,
                      dynamic_action_space=False,
                      grounded_actions_in_state=config.SHOULD_FIND_GYM_ACTION_PREDS)

        _ = env.reset()

        actions = env.get_actions()

        fringe = [env.get_state()]
        visited = set()

        progress_bar = tqdm.tqdm(unit=" states explored", position=pbar_lvl,
                                 leave=False)
        visited_states_bar = tqdm.tqdm(unit=" states visited",
                                       total=max_visited_states,
                                       position=pbar_lvl + 1,
                                       leave=False)

        while len(fringe) > 0 \
            and len(visited) < max_visited_states:

            state = fringe.pop()
            progress_bar.update(1)

            if state.literals in visited:
                continue

            visited.add(state.literals)
            visited_states_bar.update(1)

            for action in actions:

                # Reset the state since we are going
                # to try a new action from the same state.
                #
                # The simulator's internal state might have
                # changed via the step() routine from a
                # previous iteration.
                env.set_state(state)

                next_state, _, _, _ = env.step(action)
                execution_status = env.get_step_execution_status()

                action_name = action.predicate.name
                if execution_status:

                    action_sample = successful_actions.setdefault(
                        action_name, [])
                else:

                    assert next_state.literals == state.literals
                    action_sample = unsuccessful_actions.setdefault(
                        action_name, [])

                action_sample.append((state, action, next_state, execution_status))
                fringe.append(next_state)

        progress_bar.close()
        visited_states_bar.close()

    @staticmethod
    def shuffle(action_samples):

        for action, action_sample in action_samples.items():

            random.shuffle(action_sample)
    @staticmethod
    def sample(action_samples, samples_per_action):

        samples = []

        for action in action_samples:

            action_sample = action_samples[action]
            sample_count = min(len(action_sample), samples_per_action)
            samples += random.sample(action_sample, sample_count)

        return samples

    @staticmethod
    def generate_execution_based_samples(domain_file,
                                         problem_file,
                                         output_filepath,
                                         max_samples=10000,
                                         max_visited_states_per_problem=10000):

        successful_actions = dict()
        unsuccessful_actions = dict()

        env = PDDLEnv(domain_file, problem_file,
                      operators_as_actions=True,
                      raise_error_on_invalid_action=False,
                      dynamic_action_space=False,
                      grounded_actions_in_state=config.SHOULD_FIND_GYM_ACTION_PREDS)
        num_problems = len(env.problems)

        p1 = tqdm.tqdm(total=num_problems, position=0)
        for i in range(num_problems):

            VariationalDistance.generate_samples(
                domain_file, problem_file, i, successful_actions,
                unsuccessful_actions,
                max_visited_states=max_visited_states_per_problem,
                pbar_lvl=1)

            p1.update(1)

        p1.close()

        VariationalDistance.shuffle(successful_actions)
        VariationalDistance.shuffle(unsuccessful_actions)

        # Round robin selection of vd samples.
        vd_samples = []
        updated = True
        actions = successful_actions.keys() | unsuccessful_actions.keys()
        while len(vd_samples) < max_samples \
            and updated:

            updated = False
            for action in actions:
                for database in [successful_actions, unsuccessful_actions]:

                    try:
                        transitions = database.get(action, [])
                        transition = transitions.pop()
                        vd_samples.append(transition)
                        updated = True
                    except IndexError:

                        pass

        # Shuffle the transition dataset.
        random.shuffle(vd_samples)

        with open(output_filepath, "wb") as f:
            pickle.dump(vd_samples, f)

    @staticmethod
    def generate_bfs_samples(domain_file, problem_file,
                             output_filepath,
                             max_samples=3500,
                             max_bfs_explored_states=5000):
        
        action_samples = {}


        if os.path.isdir(problem_file):
            problem_files = [f for f in glob.glob(os.path.join(problem_file, "*.pddl"))]
        else:
            problem_files = [problem_file]

        p1 = tqdm.tqdm(total=len(problem_files), unit=" problems/s")
        for problem_file in problem_files:

            try:
                env = PDDLEnv(domain_file, problem_file,
                    operators_as_actions=True,
                    raise_error_on_invalid_action=True,
                    dynamic_action_space=True,
                    grounded_actions_in_state=config.SHOULD_FIND_GYM_ACTION_PREDS)
            except Exception:
                continue

            bfsagent = BFSAgent(env,
                                monte_carlo_steps=5)
            samples, _ = \
                bfsagent.generate_state_samples(
                    max_samples=max_samples // len(problem_files),
                    filter_func=VariationalDistance.FILTER_FUNC,
                    storage_func=VariationalDistance.STORAGE_FUNC)
                
            for sample in samples:

                s, a, _ = sample
                action_sample = action_samples.setdefault(a.predicate.name,
                                                          [])
                action_sample.append(sample)
            p1.update(1)

        p1.close()

        # Get the minimum count to pick for every action.
        total_count = 0
        for action_name in action_samples:

            total_count += len(action_samples[action_name])

        total_count = min(max_samples, total_count)
        assert len(action_samples) < total_count
        total_samples_to_pick = max(1, total_count // len(action_samples))

        samples = []
        for action_name in action_samples:

            action_sample = action_samples[action_name]
            num_samples = min(total_samples_to_pick, len(action_sample))

            print("Selecting", num_samples, " for", action_name)
            samples += random.sample(action_sample,  num_samples)

        if len(env.domain.actions) - len(action_samples) > 0:
            print(len(env.domain.actions) - len(action_samples),
                  " did not have any representation")
        else:
            print("All actions were represented")

        with open(output_filepath, "wb") as f:
            pickle.dump(samples, f)

if __name__ == "__main__":

    VariationalDistance.generate_execution_based_samples("Tireworld", "/tmp/results/test.pkl")
