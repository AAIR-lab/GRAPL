from curiosity_modules import create_curiosity_module
from operator_learning_modules import create_operator_learning_module
from planning_modules import create_planning_module
from pddlgym.structs import Anti
import time
import numpy as np


class Agent:
    """An agent interacts with an env, learns PDDL operators, and plans.
    This is a simple wrapper around three modules:
    1. a curiosity module
    2. an operator learning module
    3. a planning module
    The curiosity module selects actions to collect training data.
    The operator learning module learns operators from the training data.
    The planning module plans at test time.
    The planning module (and optionally the curiosity module) use the
    learned operators. The operator learning module contributes to them.
    """
    def __init__(self, domain_name, action_space, observation_space,
                 curiosity_module_name, operator_learning_name,
                 planning_module_name):
        self.curiosity_time = 0.0
        self.domain_name = domain_name
        self.curiosity_module_name = curiosity_module_name
        self.operator_learning_name = operator_learning_name
        self.planning_module_name = planning_module_name
        # The main objective of the agent is to learn good operators
        self.learned_operators = set()
        # The operator learning module learns operators. It should update the
        # agent's learned operators set
        self._operator_learning_module = create_operator_learning_module(
            operator_learning_name, self.learned_operators, self.domain_name)
        # The planning module uses the learned operators to plan at test time.
        self._planning_module = create_planning_module(
            planning_module_name, self.learned_operators, domain_name,
            action_space, observation_space)
        # The curiosity module dictates how actions are selected during training
        # It may use the learned operators to select actions
        self._curiosity_module = create_curiosity_module(
            curiosity_module_name, action_space, observation_space,
            self._planning_module, self.learned_operators,
            self._operator_learning_module, domain_name)

    ## Training time methods
    def get_action(self, state):
        """Get an exploratory action to collect more training data.
           Not used for testing. Planner is used for testing."""
        start_time = time.time()
        action = self._curiosity_module.get_action(state)
        self.curiosity_time += time.time()-start_time
        return action

    def observe(self, state, action, next_state):
        # Get effects
        effects = self._compute_effects(state, next_state)
        # Add data
        self._operator_learning_module.observe(state, action, effects)
        # Some curiosity modules might use transition data
        start_time = time.time()
        self._curiosity_module.observe(state, action, effects)
        self.curiosity_time += time.time()-start_time

    def learn(self):
        # Learn (probably less frequently than observing)
        some_operator_changed = self._operator_learning_module.learn()
        if some_operator_changed:
            start_time = time.time()
            self._curiosity_module.learning_callback()
            self.curiosity_time += time.time()-start_time
            # for pred, dt in self._operator_learning_module.learned_dts.items():
            #     print(pred)
            #     print(dt.print_conditionals())
            # print()
        # for k, v in self._operator_learning_module._ndrs.items():
        #     print(k)
        #     print(str(v))
        return some_operator_changed

    def reset_episode(self, state):
        start_time = time.time()
        self._curiosity_module.reset_episode(state)
        self.curiosity_time += time.time()-start_time

    @staticmethod
    def _compute_effects(state, next_state):
        positive_effects = {e for e in next_state.literals - state.literals}
        negative_effects = {Anti(ne) for ne in state.literals - next_state.literals}
        return positive_effects | negative_effects

    ## Test time methods
    def get_policy(self, problem_fname):
        """Get a plan given the learned operators and a PDDL problem file."""
        return self._planning_module.get_policy(problem_fname)
