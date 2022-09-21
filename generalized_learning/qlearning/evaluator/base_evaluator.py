from generalized_learning import simulator


class BaseEvaluator:

    def __init__(self, phase_dict):

        self._phase_dict = phase_dict

        self.simulator = simulator.get_simulator(
            self._phase_dict["simulator_type"])

    def evaluate(self, problem, max_time_steps):

        time_step = 0
        total_cost = 0

        current_state = problem.get_initial_state()
        done = problem.is_goal_satisfied(current_state)

        while not done \
                and time_step < max_time_steps:

            action = self.get_action(problem, current_state)

            next_state, reward, done = self.simulator.apply_action(
                problem,
                current_state,
                action)

            time_step += 1
            total_cost += reward

            if done:

                break

            current_state = next_state

        return time_step, total_cost, done
