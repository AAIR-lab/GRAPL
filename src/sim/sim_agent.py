#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import gym
from config import *
from utils import generate_ds


class SimAgent(object):
    def __init__(self, pddl_gym_env_name, pddl_domain_name, custom_fd):
        """
        Generic agent for pddlgym environments
        creates the environment and parses domain files,
        """

        self.new_problem_number = 99
        self.action_mapping = {}
        self.pddl_gym_env_name = pddl_gym_env_name
        self.pddl_domain_name = pddl_domain_name
        self.custom_fd = custom_fd
        self.random_states = []
        try:
            os.remove(self.custom_fd)
        except OSError as e:
            print(e)
            pass
        self.n = 0
        self.env = gym.make(pddl_gym_env_name)
        self.state, dbg = self.env.reset()
        self.env_actions = []
        self.problem_file_loaded = dbg['problem_file']
        for action in list(self.env.action_space.all_ground_literals()):
            self.env_actions.append(action.pddl_str()[1:-1])
        self.action_parameters, self.predicates, _, _, self.objects, self.types, _, _ = generate_ds(
            self.env.domain.domain_fname, self.env.problems[0].problem_fname)

    def print_model(self):
        print("\n ----PDDLGYM Model----:\n " + str(self.pddl_domain_name) + "\n Actions:" + str(
            self.env_actions) + "\n current state: " + str(self.state))

    @staticmethod
    def state_rep_convert(state):
        state = list(state)
        for i in range(len(state)):
            state[i] = '|'.join(state[i].pddl_str()[1:-1].split(' '))
        return state

    def test_reset(self, set_state=False, remake=True, set_problem_file=None):
        """
        env.reset() only works if all the files defines within the directory for the selected
        environment are syntactically valid
        This can be used to check if the custom initial state supplied by the query has been
        correctly written to the pddl file.
        This method checks if all files are valid by performing an environment reset
        Returns True for success and False for Failure
        """

        self.env.close()
        try:
            if remake:
                self.env = gym.make(self.pddl_gym_env_name)
            if set_problem_file is not None:
                self.env.fix_problem_index(set_problem_file)

            st, debug_info = self.env.reset(rend=False)
            if set_state:
                self.state = st
            if VERBOSE:
                print("Problem File 1:" + str(debug_info['problem_file']))
            self.problem_file_loaded = debug_info['problem_file']
            return True
        except Exception as e:
            print("Couldn't reset environement! Error:" + str(e))
            return False

    def render_sim(self):
        """
        returns a render of the current state of the environment
        """

        return self.env.render()

    def load_custom_problemfile(self):
        """
        loads the custom problem file;
        To be used after setting initial state to create env with custom initial state as
        specifies in query

        """

        self.n += 1
        if VERBOSE:
            print("Loading custom problem file")
        for i, p in enumerate(self.env.problems):
            if str(self.new_problem_number) in p.problem_fname.split('/')[-1]:
                self.env.fix_problem_index(i)
                if VERBOSE:
                    print("Fixed problem file")
                self.state, dbg = self.env.reset()
                if VERBOSE:
                    print("reset")
                self.problem_file_loaded = dbg['problem_file']
                if VERBOSE:
                    print("Loaded custom problem : " + str(p.problem_fname))
                return
        if VERBOSE:
            print("Couldn't load custom problem!")
        return
