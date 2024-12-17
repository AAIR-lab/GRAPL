
from curiosity_modules.goal_babbling import GoalBabblingCuriosityModule
from settings import AgentConfig as ac
from pddlgym import structs
from pddlgym.inference import find_satisfying_assignments

from collections import defaultdict
import copy
import itertools
import numpy as np


class GLIBLCuriosityModule(GoalBabblingCuriosityModule):

    _k = None # Must be set by subclasses
    _ignore_statics = True
    _ignore_mutex = True

    ### Initialization ###

    def _initialize(self):
        super()._initialize()
        self._rand_state = np.random.RandomState(seed=ac.seed)
        self._name = "newiw"
        self._episode_start_state = None

        # Initialize all possible goal-actions up to the max num of lits.
        # This is a set of (goal literals, action literal) pairs.
        self._unseen_goal_actions = self._init_unseen_goal_actions(
            self._action_space.predicates, self._observation_space.predicates, 
            self._k)

    @classmethod
    def _use_goal_preds(cls, goal_preds):
        return True

    @classmethod
    def _init_unseen_goal_actions(cls, action_predicates, observation_predicates, max_num_lits):
        """Initialize all possible goal-actions up to the max num of lits.
        Parameters
        ----------
        action_predicates : { Predicate }
        observation_predicates : { Predicate }
        max_num_lits : max_num_lits
        Returns
        -------
        unseen_goal_actions : (tuple(Literal), Literal)
            Pairs of (goal literals, action).
        """
        unseen_goal_actions = set()
        for action_pred in action_predicates:
            for num_lits in range(1, max_num_lits+1):
                for goal_preds in itertools.combinations(observation_predicates, num_lits):
                    # Lump together goal and action for now; will separate at the end
                    # if num_lits == 1:
                    #     if not goal_preds[0]=="robotat":continue
                    # else:
                    #     if not (goal_preds[1] == "robotat" and goal_preds[0].name.startswith("locprize")):continue
                    if not cls._use_goal_preds(goal_preds):
                        continue
                    goal_action_preds = list(goal_preds) + [action_pred]
                    # Find all possible variable assortments for the goal action predicates
                    # We're going to first create unique placeholders for the slots in the predicates
                    ph_to_pred_slot = {}
                    goal_action_lits_with_phs = []
                    # Create the placeholders
                    for i, pred in enumerate(goal_action_preds):
                        ph_for_lit = []
                        for j, var_type in enumerate(pred.var_types):
                            ph = var_type('ph{}_{}'.format(i, j))
                            ph_to_pred_slot[ph] = (i, j)
                            ph_for_lit.append(ph)
                        ph_lit = pred(*ph_for_lit)
                        goal_action_lits_with_phs.append(ph_lit)
                    phs = sorted(ph_to_pred_slot.keys())
                    # Consider all substitutions of placeholders to variables
                    for vs in cls._iter_vars_from_phs(phs):
                        goal_vs = {v.name for i, v in enumerate(vs) if
                                   ph_to_pred_slot[phs[i]][0] != len(goal_action_preds)-1}
                        action_vs = {v.name for i, v in enumerate(vs) if
                                     ph_to_pred_slot[phs[i]][0] == len(goal_action_preds)-1}
                        if goal_vs and action_vs-goal_vs and min(action_vs-goal_vs) < max(goal_vs):
                            continue
                        goal_action_lits = [copy.deepcopy(lit) for lit in goal_action_lits_with_phs]
                        # Perform substitution
                        for k, v in enumerate(vs):
                            ph = phs[k]
                            (i, j) = ph_to_pred_slot[ph]
                            goal_action_lits[i].update_variable(j, v)
                        # Goal lits cannot have repeated vars
                        goal_action_valid = True
                        for lit in goal_action_lits:
                            if len(set(lit.variables)) != len(lit.variables):
                                goal_action_valid = False
                                break
                        # Finish the goal and add it
                        if goal_action_valid:
                            goal = tuple([l for l in goal_action_lits if l.predicate != action_pred])
                            action = [l for l in goal_action_lits if l.predicate == action_pred][0]
                            unseen_goal_actions.add((goal, action))
        return unseen_goal_actions

    ### Reset ###

    def _iw_reset(self):
        # Want to retry goal-actions if a new episode or new operators learned
        self._untried_episode_goal_actions = copy.deepcopy(self._unseen_goal_actions)
        # Randomly shuffle within num_lits
        self._untried_episode_goal_actions = sorted(self._untried_episode_goal_actions,
            key=lambda ga : (len(ga[0]), self._rand_state.uniform()))
        if self._ignore_statics:  # ignore static goals
            static_preds = self._compute_static_preds()
            self._untried_episode_goal_actions = list(filter(
                lambda ga: any(lit.predicate not in static_preds for lit in ga[0]),
                self._untried_episode_goal_actions))
        if self._ignore_mutex:  # ignore mutex goals
            mutex_pairs = self._compute_lifted_mutex_literals(self._episode_start_state)
            self._untried_episode_goal_actions = list(filter(
                lambda ga: frozenset(ga[0]) not in mutex_pairs,
                self._untried_episode_goal_actions))
        # Forget the goal-action that was going to be taken at the end of the plan in progress
        self._current_goal_action = None

    def reset_episode(self, state):
        super().reset_episode(state)
        self._episode_start_state = state
        self._iw_reset()

    def learning_callback(self):
        super().learning_callback()
        self._iw_reset()

    def _get_fallback_action(self, state):
        self._current_goal_action = None
        return super()._get_fallback_action(state)

    ### Get an action ###

    def _get_action(self, state):
        # First check whether we just finished a plan and now must take the final action
        if (not (self._current_goal_action is None)) and (len(self._plan) == 0):
            action = self._get_ground_action_to_execute(state)
            if action != None:
                # print("*** Finished the plan, now executing the action", action)
                # Execute the action
                self.line_stats.append(1)
                return action
        # Either continue executing a plan or make a new one (or fall back to random)
        return super()._get_action(state)

    def _get_ground_action_to_execute(self, state):
        lifted_goal, lifted_action = self._current_goal_action
        # Forget this goal-action because we're about to execute it
        self._current_goal_action = None
        # Sample a grounding for the action conditioned on the lifted goal and state
        action = self._sample_action_from_goal(lifted_goal, lifted_action, state, self._rand_state)
        # If the action is None, that means that the plan was wrong somehow.
        return action

    @staticmethod
    def _sample_action_from_goal(lifted_goal, lifted_action, state, rand_state):
        """Sample a grounding for the action conditioned on the lifted goal and state"""
        # Try to find a grounding of the lifted goal in the state
        all_assignments = find_satisfying_assignments(state.literals, lifted_goal,
            allow_redundant_variables=False)
        # If none exist, return action None
        if len(all_assignments) == 0:
            return None
        assignments = all_assignments[0]
        # Sample an action conditioned on the assignments.
        # Find possible groundings for each object by type.
        types_to_objs = defaultdict(set)
        for lit in state.literals:
            for obj in lit.variables:
                types_to_objs[obj.var_type].add(obj)
        # Sample a grounding for all the unbound variables.
        grounding = []
        for v in lifted_action.variables:
            if v in assignments:
                # Variable is ground, so go with ground variable.
                grounding.append(assignments[v])
            else:
                # Sample a grounding. Make sure it's not an assigned value.
                choices = set(types_to_objs[v.var_type])
                choices -= set(assignments.values())
                choices -= set(grounding)
                # There's no way to bind the variables of the action.
                if len(choices) == 0:
                    return None
                choice = sorted(choices)[rand_state.choice(len(choices))]
                grounding.append(choice)
        assert len(grounding) == len(set(grounding))
        return lifted_action.predicate(*grounding)

    def _sample_goal(self, state):
        """Produce a new goal to try to plan towards"""
        # Note that these are already in random order as per _iw_reset
        if len(self._untried_episode_goal_actions) > 0:
            goal, action = self._untried_episode_goal_actions.pop(0)
            # print("sampling goal-action", goal,action)
            self._current_goal_action = (goal, action)
            return self._structify_goal(goal)
        # No goals left to try
        # print("no goals left")
        return None

    def _finish_plan(self, plan):
        # If the plan is empty, then we want to immediately take the action.
        if len(plan) == 0:
            action = self._get_ground_action_to_execute(self._last_state)
            # print("Goal is satisfied in the current state; taking action now:", action)
            if action is None:
                # There was no way to bind the lifted action. Fallback
                action = self._get_fallback_action(self._last_state)
            return [action]
        # Otherwise, we'll take the last action once we finish the plan
        # print("Setting a plan:", plan)
        return plan

    def _goal_is_valid(self, goal):
        return not (goal is None)

    def _plan_is_good(self):
        return True

    @staticmethod
    def _structify_goal(goal):
        """Create Exists struct for a goal."""
        variables = sorted({ v for lit in goal for v in lit.variables })
        body = structs.LiteralConjunction(goal)
        return structs.Exists(variables, body)

    ### Update caches based on new observation ###

    def observe(self, state, action, _effects):
        # Find goal-actions that we're seeing for the first time
        newly_seen_goal_actions = []
        for goal_action in self._unseen_goal_actions:
            # Check whether goal-action is satisfied by new state and action
            conds = list(goal_action[0]) + [goal_action[1]]
            assignments = find_satisfying_assignments(state.literals | {action}, conds,
                allow_redundant_variables=False)
            if len(assignments) > 0:
                newly_seen_goal_actions.append(goal_action)
        for new_goal_action in newly_seen_goal_actions:
            # print("Removing goal-action:", new_goal_action)
            self._unseen_goal_actions.remove(new_goal_action)


class GLIBL2CuriosityModule(GLIBLCuriosityModule):
    _k = 2
