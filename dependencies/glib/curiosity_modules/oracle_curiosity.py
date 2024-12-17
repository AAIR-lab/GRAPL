"""Oracle curiosity that uses knowledge of the true operators.
Should be an upper bound on other methods."""

import numpy as np
from curiosity_modules import BaseCuriosityModule
from settings import AgentConfig as ac
from pddlgym import structs


class OracleCuriosityModule(BaseCuriosityModule):
    def _initialize(self):
        self._name = "oracle"
        self._rand_state = np.random.RandomState(seed=ac.seed)
        self._num_steps = 0
        # Keep track of the number of times that we lookahead
        self.lookaheads = []
        # Keep track of the number of times that we do goal-directed search
        self.goaldirecteds = []
        # Keep track of the number of times that we fall back to random
        self.fallbacks = []
        self._oracle_on = True

    def reset_episode(self, state):
        self._last_state = set()
        self._plan = []

    def turn_off(self):
        self._oracle_on = False

    def get_action(self, state):
        action = self._get_action(state, depth=0, max_depth=ac.oracle_max_depth)[0]
        self._num_steps += 1
        return action

    def _get_action(self, state, depth, max_depth):
        """Returns tuple of (action, is_interesting).
        """
        if self._domain_name == "PybulletBlocks":
            return self._action_space.sample(state), False
        if ac.learning_name == "TILDE":
            action_preds_with_learned_rules = set(self._operator_learning_module.learned_dts)
        elif ac.learning_name == "LNDR":
            action_preds_with_learned_rules = set(self._operator_learning_module._ndrs)
        else:
            import ipdb; ipdb.set_trace()

        assert max_depth > 0

        if not self._oracle_on:
            self.lookaheads.append(0)
            self.goaldirecteds.append(0)
            self.fallbacks.append(1)
            # print("Sampling random action")
            return self._action_space.sample(state), False

        if depth == 0:
            last_state = self._last_state
            self._last_state = state
            if self._plan and (last_state != state):
                # We have an unfinished plan, keep executing it.
                self.lookaheads.append(0)
                self.goaldirecteds.append(1)
                self.fallbacks.append(0)
                # print("Continuing plan", self._plan)
                return self._plan.pop(0), True

        if depth == max_depth:
            # print("Sampling random action")
            return self._action_space.sample(state), False

        all_ground_actions = sorted(self._action_space.all_ground_literals(state))
        self._rand_state.shuffle(all_ground_actions)
        for action in all_ground_actions:
            # If no decision tree learned for this action predicate,
            # it's automatically interesting.
            if not any(act_pred.name == action.predicate.name
                       for act_pred in action_preds_with_learned_rules):
                assert depth == 0
                self.lookaheads.append(0)
                self.goaldirecteds.append(0)
                self.fallbacks.append(0)
                # print("found interesting action (1):", action)
                return action, True

            if self._is_goal_state_action(state, action):
                if depth == 0:
                    self.lookaheads.append(0)
                    self.goaldirecteds.append(0)
                    self.fallbacks.append(0)
                # print("found interesting action (2):", action)
                return action, True

        # All learned operators are perfect for the current state. So let's do
        # some recursive lookahead...
        self._plan = self._bfs(state)
        if len(self._plan) > 0:
            self.lookaheads.append(1)
            self.goaldirecteds.append(0)
            self.fallbacks.append(0)
            # print("Found a new plan, taking first action:", self._plan)
            return self._plan.pop(0), True

        # Give up and sample a random action.
        if depth == 0:
            self.lookaheads.append(0)
            self.goaldirecteds.append(0)
            self.fallbacks.append(1)
        # print("Taking a random action")
        return self._action_space.sample(state), False

    def _is_goal_state_action(self, state, action):
        """A state-action is a goal if the predicted next state is different
           from the ground truth."""
        # Calculate predicted next state under learned operators.
        predicted_next_state = self._get_predicted_next_state(state, action)
        actual_next_state = self._predict_ground_truth(state, action)
        return predicted_next_state != actual_next_state

    def _predict_ground_truth(self, state, action):
        # Save current operators.
        old_ops = set()
        for op in self._learned_operators:
            old_ops.add(op)
        # Cheat by hacking in ground truth operators.
        self._learned_operators.clear()
        for op in ac.train_env.domain.operators.values():
            self._learned_operators.add(op)
        # Check for operators as actions
        for op in self._learned_operators:
            if not any(p.predicate in ac.train_env.domain.actions for p in op.preconds.literals):
                assert op.name in ac.train_env.domain.actions
                action_predicate = [p for p in ac.train_env.domain.actions if p.name == op.name][0]
                op.preconds.literals.append(action_predicate(*op.params))
        # Calculate actual next state under ground truth operators.
        actual_next_state = self._get_predicted_next_state_ops(state, action, mode="max")
        # Restore current operators.
        self._learned_operators.clear()
        for op in old_ops:
            self._learned_operators.add(op)
        return actual_next_state


    def _bfs(self, init_state):
        """Graph search to compute all H-step novelties.
        """
        queue = [([init_state], [])]
        # Run BFS.
        while queue:
            node = queue.pop(0)
            path, act_seq = node
            state = path[-1]
            if state is None or len(act_seq) == ac.oracle_max_depth:
                continue
            for action in self._action_space.all_ground_literals(state):
                if self._is_goal_state_action(state, action):
                    return act_seq+[action]
                predicted_next_state = self._predict_ground_truth(state, action)
                queue.append((path+[predicted_next_state], act_seq+[action]))
        return []
