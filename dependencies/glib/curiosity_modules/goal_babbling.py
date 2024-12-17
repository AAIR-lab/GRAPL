"""Curiosity module that samples previously unachieved goals and plans to
achieve them with the current operators.
"""

import os
from planning_modules.base_planner import PlannerTimeoutException, \
    NoPlanFoundException
from settings import AgentConfig as ac
from curiosity_modules import BaseCuriosityModule


class GoalBabblingCuriosityModule(BaseCuriosityModule):
    """Curiosity module that samples a completely random literal and plans to
    achieve it with the current operators.
    """
    def _initialize(self):
        self._num_steps = 0
        self._name = "goalbabbling"
        # Keep track of the number of times that we follow a plan
        self.line_stats = []

    def reset_episode(self, state):
        self._last_state = set()
        self._plan = []

    def _sample_goal(self, state):
        return self._observation_space.sample_literal(state)

    def _goal_is_valid(self, goal):
        return True

    def _finish_plan(self, plan):
        return plan

    def get_action(self, state):
        """Execute plans open loop until stuck, then replan"""
        action = self._get_action(state)
        self._num_steps += 1
        return action

    def _get_action(self, state):
        last_state = self._last_state
        self._last_state = state

        # Continue executing plan?
        if self._plan and (last_state != state):
            self.line_stats.append(1)
            # print("CONTINUING PLAN")
            return self._plan.pop(0)

        # Try to sample a goal for which we can find a plan
        sampling_attempts = planning_attempts = 0
        while (sampling_attempts < ac.max_sampling_tries and \
               planning_attempts < ac.max_planning_tries):

            goal = self._sample_goal(state)
            sampling_attempts += 1

            if not self._goal_is_valid(goal):
                continue

            # print("trying goal:",goal)

            # Create a pddl problem file with the goal and current state
            problem_fname = self._create_problem_pddl(
                state, goal, prefix=self._name)

            # Get a plan
            try:
                self._plan = self._planning_module.get_plan(
                    problem_fname, use_cache=False)
            except NoPlanFoundException:
                os.remove(problem_fname)
                continue
            except PlannerTimeoutException:
                os.remove(problem_fname)
                break
            os.remove(problem_fname)
            planning_attempts += 1

            if self._plan_is_good():
                self._plan = self._finish_plan(self._plan)
                print("\tGOAL:", goal)
                print("\tPLAN:", self._plan)
                # import ipdb; ipdb.set_trace()
                # Take the first step in the plan
                self.line_stats.append(1)
                return self._plan.pop(0)
            self._plan = []

        # No plan found within budget; take a random action
        # print("falling back to random")
        return self._get_fallback_action(state)

    def _get_fallback_action(self, state):
        self.line_stats.append(0)
        return self._action_space.sample(state)

    def _plan_is_good(self):
        return bool(self._plan)
