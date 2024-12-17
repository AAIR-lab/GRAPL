import gym
import pddlgym

class GroundTruthOperatorLearningModule:
    def __init__(self, env_name, learned_operators):
        env = gym.make(env_name)
        for operator in env.domain.operators.values():
            learned_operators.add(operator)
        self._changed = True

    def observe(self, state, action, effects):
        pass

    def learn(self):
        changed = self._changed
        self._changed = False
        return changed
