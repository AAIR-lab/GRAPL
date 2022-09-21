
import os
import pathlib
import subprocess

from generalized_learning.util import file
from qlearning.results import QLearningResults


class TrapsNet:

    DEFAULTS = {
        "episodes_per_epoch": 25,
        "num_episodes": 250,
        "timesteps_per_episode": 40,
        "gamma": 1.0,
        "model_dir": None,
        "debug_print": False,
        "num_greedy_evaluations_per_episode": 10,
        "greedy_evaluation_epsilon": 0.0,
        "skip_leapfrog": False,
        "parallelism": 1,
        "keep_log_file": True,
    }

    TRAPSNET_SCRIPT = os.path.abspath((pathlib.Path(
        __file__).parent / "../../scripts/trapsnet_runner.sh").as_posix())

    def _get_value(self, config_dict, key):

        try:

            return config_dict[key]
        except KeyError:

            return TrapsNet.DEFAULTS[key]

    def __init__(self, parent_dir, phase_dict):

        self._parent_dir = parent_dir
        self._phase_dict = phase_dict

        self.reset()

    def reset(self):

        assert "experiment_name" in self._phase_dict
        self._experiment_name = self._get_value(
            self._phase_dict,
            "experiment_name")

        self._num_episodes = self._get_value(
            self._phase_dict,
            "num_episodes")

        self._episodes_per_epoch = self._get_value(
            self._phase_dict,
            "episodes_per_epoch")
        assert self._num_episodes % self._episodes_per_epoch == 0

        self._timesteps_per_episode = self._get_value(
            self._phase_dict,
            "timesteps_per_episode")

        self._gamma = self._get_value(
            self._phase_dict,
            "gamma")

        self._restore_dir = self._get_value(
            self._phase_dict,
            "model_dir")

        if self._restore_dir is not None:
            self._restore_dir = file.get_relative_path(
                self._restore_dir,
                self._parent_dir)

        self._debug_print = self._get_value(
            self._phase_dict,
            "debug_print")

        self._num_greedy_evaluations_per_episode = self._get_value(
            self._phase_dict,
            "num_greedy_evaluations_per_episode")

        self._greedy_evaluation_epsilon = self._get_value(
            self._phase_dict,
            "greedy_evaluation_epsilon")

        self._skip_leapfrog = self._get_value(
            self._phase_dict,
            "skip_leapfrog")

        self._parallelism = self._get_value(
            self._phase_dict,
            "parallelism")

        self._keep_log_file = self._get_value(
            self._phase_dict,
            "keep_log_file")

    def _plot(self, input_dir):

        WRITE_MODE = "a"

        cost_output_prefix = "%s_episode_cost" % (self._experiment_name)
        epoch_output_prefix = "%s_epoch_cost" % (self._experiment_name)
        epoch_rate_prefix = "%s_epoch_rate" % (self._experiment_name)

        cost_results = QLearningResults(
            input_dir,
            fieldnames=["problem_name", "experiment_name",
                        "(episode, avg_cost)"],
            output_prefix=cost_output_prefix,
            write_mode=WRITE_MODE)

        episode_results = QLearningResults(
            input_dir,
            fieldnames=["problem_name", "experiment_name",
                        "(epoch_no, avg_cost)"],
            output_prefix=epoch_output_prefix,
            write_mode=WRITE_MODE)

        cost_results.plot(input_dir, "Episode #", "Avg. episode cost",
                          lambda x: x[0],
                          lambda x: x[1])

        episode_results.plot(input_dir, "Epoch", "Success rate",
                             lambda x: x[0],
                             lambda x: x[1],
                             output_prefix=epoch_rate_prefix)

        episode_results.plot(input_dir, "Epoch", "Avg. cost",
                             lambda x: x[0],
                             lambda x: x[2])

    def run(self, input_dir):

        cmd_string = "%s" % (TrapsNet.TRAPSNET_SCRIPT)

        cmd_string += " --base_dir %s" % (input_dir)
        cmd_string += " --num_episodes %s" % (self._num_episodes)
        cmd_string += " --timesteps_per_episode %s" % (
            self._timesteps_per_episode)

        cmd_string += " --episodes_per_epoch %s" % (self._episodes_per_epoch)
        cmd_string += " --parallelism %s" % (self._parallelism)
        cmd_string += " --experiment_name %s" % (self._experiment_name)
        cmd_string += " --gamma %s" % (self._gamma)

        if self._restore_dir is not None:

            cmd_string += " --restore_dir %s" % (self._restore_dir)

        log_filepath = "%s/%s.log" % (input_dir, self._experiment_name)
        log_file_handle = open(log_filepath, "w")

        if self._debug_print:

            print("Executing: %s" % (cmd_string))

        trapsnet_cmd = cmd_string.split(" ")
        completed_process = subprocess.Popen(
            trapsnet_cmd,
            stdout=log_file_handle,
            stderr=log_file_handle)

        completed_process.wait()
        log_file_handle.close()

        if not self._keep_log_file:

            os.remove(log_filepath)

        # Trapsnet should already setup the qlearning results in a form
        # that is ready to be plotted.
        self._plot(input_dir)
