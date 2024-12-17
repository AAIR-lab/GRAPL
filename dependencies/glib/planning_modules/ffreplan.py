from planning_modules.base_planner import Planner, PlannerTimeoutException, \
    NoPlanFoundException
from planning_modules.ff import FastForwardPlanner

from pddlgym.inference import find_satisfying_assignments
from pddlgym.parser import parse_plan_step, PDDLProblemParser
from pddlgym import structs
from settings import AgentConfig as ac

import sys
import os
import re
import subprocess
import time
import tempfile


class FFReplanner(Planner):

    def __init__(self, learned_operators, domain_name, action_space, observation_space):
        super().__init__(learned_operators, domain_name, action_space, observation_space)
        self._ff_planner = FastForwardPlanner(self._learned_operators, self.domain_name, 
            self._action_space, self._observation_space)

    def get_policy(self, raw_problem_fname):
        # Only replan if outcome is not expected
        expected_next_state = None
        plan = []

        # Given a state, replan and execute first section in plan
        def policy(obs):
            nonlocal expected_next_state
            nonlocal plan

            state = obs.literals
            goal = obs.goal
            objects = obs.objects

            # Add possible actions to state
            full_state = set(state)
            full_state.update(self._action_space.all_ground_literals(obs))

            # Create problem file
            file = tempfile.NamedTemporaryFile(delete=True)
            fname = file.name
            PDDLProblemParser.create_pddl_file(fname, objects, full_state, "myproblem", self.domain_name, goal)
            # Get plan
            plan = self._ff_planner.get_plan(fname, use_cache=False)
            # Updated expected next state
            expected_next_state = self._get_predicted_next_state_ops(obs, plan[0])
            return plan.pop(0)

        return policy

    def get_plan(self, raw_problem_fname, **kwargs):
        return self._ff_planner.get_plan(raw_problem_fname, **kwargs)

    # TODO: The three methods below are copied from curiosity_base. Refactor.
    def _get_predicted_next_state_ops(self, state, action):
        """
        """
        for op in self._learned_operators:
            assignments = self._preconds_satisfied(state, action, op.preconds.literals)
            if assignments is not None:
                ground_effects = [structs.ground_literal(l, assignments) \
                    for l in op.effects.literals]
                return self._execute_effects(state, ground_effects)
        return state  # no change

    @staticmethod
    def _preconds_satisfied(state, action, literals):
        """Helper method for _get_predicted_next_state.
        """
        kb = state.literals | {action}
        assignments = find_satisfying_assignments(kb, literals)
        # NOTE: unlike in the actual environment, here num_found could be
        # greater than 1. This is because the learned operators can be
        # wrong by being overly vague, so that multiple assignments get
        # found. Here, if num_found is greater than 1, we just use the first.
        if len(assignments) == 1:
            return assignments[0]
        return None

    @staticmethod
    def _execute_effects(state, literals):
        """Helper method for _get_predicted_next_state.
        """
        new_literals = set(state.literals)
        for effect in literals:
            if effect.predicate.name == "NoChange":
                continue
            # Negative effect
            if effect.is_anti:
                literal = effect.inverted_anti
                if literal in new_literals:
                    new_literals.remove(literal)
        for effect in literals:
            if effect.predicate.name == "NoChange":
                continue
            if not effect.is_anti:
                new_literals.add(effect)
        return state.with_literals(new_literals)
