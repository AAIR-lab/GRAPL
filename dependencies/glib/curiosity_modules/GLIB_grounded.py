"""Goal-literal babbling with grounded novelty. Outputs single-literal goals and
also actions.
"""

import numpy as np
from settings import AgentConfig as ac
from curiosity_modules.goal_babbling import GoalBabblingCuriosityModule


class GLIBG1CuriosityModule(GoalBabblingCuriosityModule):
    _ignore_statics = True

    def _initialize(self):
        self._num_steps = 0
        self._rand_state = np.random.RandomState(seed=ac.seed)
        self._name = "glibg1"
        self._static_preds = self._compute_static_preds()
        # Keep track of the number of times that we follow a plan
        self.line_stats = []

    def reset_episode(self, state):
        self._recompute_unseen_lits_acts(state)
        self._last_state = set()
        self._plan = []

    def _recompute_unseen_lits_acts(self, state):
        self._unseen_lits_acts = set()
        for lit in self._observation_space.all_ground_literals(state):
            if self._ignore_statics and \
               lit.predicate in self._static_preds:  # ignore static goals
                continue
            for act in self._action_space.all_ground_literals(state):
                self._unseen_lits_acts.add((lit, act))
        self._unseen_lits_acts = sorted(self._unseen_lits_acts)

    def _get_action(self, state):
        if self._unseen_lits_acts is None:
            self._recompute_unseen_lits_acts(state)
        action = super()._get_action(state)
        for lit in state:  # update novelty
            if (lit, action) in self._unseen_lits_acts:
                self._unseen_lits_acts.remove((lit, action))
        return action

    def learning_callback(self):
        super().learning_callback()
        self._static_preds = self._compute_static_preds()
        self._unseen_lits_acts = None

    def _sample_goal(self, state):
        if not self._unseen_lits_acts:
            return None
        goal, act = self._unseen_lits_acts[self._rand_state.choice(
            len(self._unseen_lits_acts))]
        self._last_sampled_action = act
        return goal

    def _goal_is_valid(self, goal):
        return not (goal is None)

    def _finish_plan(self, plan):
        self._last_state = None
        return plan + [self._last_sampled_action]
