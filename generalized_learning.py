"""
generalized_learning.py
========================

main() app.
"""


def update_pythonpath():

    import pathlib
    import sys

    root = pathlib.Path(__file__).parent

    sys.path.append((root / "generalized_learning").as_posix())

    fd_root_path = root / "dependencies" / "fast-downward-682f00955a82"
    sys.path.append(fd_root_path.as_posix())
    sys.path.append((fd_root_path / "src" / "translate").as_posix())
    sys.path.append((fd_root_path / "driver").as_posix())


def ignore_warnings():

    import logging
    import warnings
    warnings.simplefilter("ignore", FutureWarning)

    from yaml import YAMLLoadWarning
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf

    warnings.simplefilter("ignore", YAMLLoadWarning)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)


update_pythonpath()
ignore_warnings()

import argparse
import copy
import datetime
import json
import logging
import pathlib
import shutil
import subprocess

from abstraction.model import Model
from benchmarks.generator import Generator
from plot.plot import Plot
from search.solver import Solver
from util import constants
from util import phase
from util.phase import Phase
from util.phase import PhaseManager

logger = logging.getLogger(__name__)


class Leapfrog(Phase):

    REQUIRED_KEYS = set(["num_loops"]).union(Phase.REQUIRED_KEYS)

    #: The default phase dict for this phase.
    DEFAULT_PHASE_DICT = Phase.DEFAULT_PHASE_DICT

    @staticmethod
    def get_instance(parent, parent_dir, global_dict, user_phase_dict,
                     failfast):

        return Leapfrog(parent, parent_dir, global_dict,
                        user_phase_dict, failfast)

    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        super(Leapfrog, self).__init__(parent, parent_dir, global_dict,
                                       user_phase_dict, failfast)

        # Override the base_dir with the parent_dir.
        # We don't care for the leapfrogger to create a new directory.
        self._base_dir = self._parent_dir

    def _get_phase_dict_instance(self, loop_no, phase_dict):

        new_phase_dict = {}

        # Fix up all the list values with the loop.
        for key in phase_dict.keys():

            if isinstance(phase_dict[key], list):

                value_list = phase_dict[key]
                try:

                    if value_list[loop_no] is not None:
                        new_phase_dict[key] = value_list[loop_no]
                except IndexError:

                    # Set it to the last value on index error.
                    assert value_list[-1] is not None
                    new_phase_dict[key] = value_list[-1]
            else:

                assert phase_dict[key] is not None
                new_phase_dict[key] = phase_dict[key]

        return new_phase_dict

    def _get_generator_phase_dict(self, loop_no, old_dict):

        current_name = self.get_name(loop_no)
        keys = old_dict.keys()

        new_dict = self._get_phase_dict_instance(
            loop_no,
            old_dict)

        # Start filling in the blanks.
        assert "name" not in keys
        new_dict["name"] = current_name

        return new_dict

    def _get_trainer_phase_dict(self, loop_no, old_dict):

        current_name = self.get_name(loop_no)
        keys = old_dict.keys()

        new_dict = self._get_phase_dict_instance(
            loop_no,
            old_dict)

        assert "name" not in keys
        new_dict["name"] = current_name

        # Start filling in the blanks.
        assert "model_name" in keys

        assert "input_dir" not in keys
        new_dict["input_dir"] = current_name

        if "input_dir" not in keys:

            new_dict["input_dir"] = current_name

        if loop_no > 0:

            # Use the previous network to train the current network.
            new_dict["model_dir"] = self.get_name(loop_no - 1)

        new_dict["nodes_expanded_limit"] = 30000
        return new_dict

    def _get_evaluator_phase_dict(self, loop_no, old_dict):

        current_name = self.get_name(loop_no)
        keys = old_dict.keys()

        new_dict = self._get_phase_dict_instance(
            loop_no,
            old_dict)

        assert "name" not in keys
        new_dict["name"] = current_name

        # Start filling in the blanks.
        assert "model_name" in keys
        assert "input_dir" in keys

        new_dict["model_dir"] = self.get_name(loop_no)

        return new_dict

    def _get_model_phase_dict(self, loop_no, old_dict):

        current_name = self.get_name(loop_no)
        keys = old_dict.keys()

        new_dict = self._get_phase_dict_instance(
            loop_no,
            old_dict)

        # Start filling in the blanks.
        assert "nn_type" in keys
        assert "nn_name" in keys

        assert "name" not in keys
        assert "solver_name" not in keys
        assert "input_dir" not in keys

        new_dict["name"] = current_name
        new_dict["solver_name"] = current_name
        new_dict["input_dir"] = current_name

        return new_dict

    def _get_phase(self, loop_no, phase_key, phase_dict):

        if "generator" == phase_key:

            generator_dict = self._get_generator_phase_dict(
                loop_no, phase_dict)
            return Generator.get_instance(self, self._base_dir,
                                          self._global_dict, generator_dict,
                                          self._failfast)
        elif "trainer" == phase_key:

            trainer_dict = self._get_trainer_phase_dict(loop_no, phase_dict)
            return Solver.get_instance(self, self._base_dir, self._global_dict,
                                       trainer_dict, self._failfast)
        elif "model" == phase_key:

            model_dict = self._get_model_phase_dict(loop_no, phase_dict)
            return Model.get_instance(self, self._base_dir, self._global_dict,
                                      model_dict, self._failfast)
        elif "evaluator" == phase_key:

            evaluator_dict = self._get_evaluator_phase_dict(loop_no,
                                                            phase_dict)
            return Solver.get_instance(self, self._base_dir, self._global_dict,
                                       evaluator_dict, self._failfast)
        else:

            assert False
            return None

    def get_name(self, loop_no):

        return "l%u" % (loop_no)

    def execute(self, *args, **kwargs):

        (args)
        (kwargs)

        results = []

        num_loops = self._phase_dict["num_loops"]

        for loop_no in range(num_loops):

            for phase_dict in self._phase_dict["phases"]:

                assert len(phase_dict) == 1
                phase_key = next(iter(phase_dict))
                phase_dict = phase_dict[phase_key]

                logger.info("Running leapfrog, name=%s, loop=%u, phase=%s" %
                            (self._phase_dict["name"], loop_no, phase_key))

                phase = self._get_phase(loop_no, phase_key, phase_dict)

                if phase.can_execute():

                    results.append(phase.execute())

        return results


class Config(Phase):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Besides the default keys, we also need "type" to be specified.
    REQUIRED_KEYS = set(["name"]).union(Phase.REQUIRED_KEYS)

    #: The default phase dict for this phase.
    DEFAULT_PHASE_DICT = Phase.DEFAULT_PHASE_DICT

    @staticmethod
    def get_instance(parent, parent_dir, global_dict, user_phase_dict,
                     failfast):

        return Phase.get_instance(parent, parent_dir, global_dict,
                                  user_phase_dict, failfast)

    def __init__(self, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        # Pass parent=None, since config is the root of the hierarchy.
        super(Config, self).__init__(None, parent_dir, global_dict,
                                     user_phase_dict, failfast)

        self.initialize_directories()

    def _get_phase(self, phase_key, phase_dict):

        if "generator" == phase_key:

            return Generator.get_instance(self, self._base_dir,
                                          self._global_dict, phase_dict,
                                          self._failfast)
        elif "solver" == phase_key:

            return Solver.get_instance(self, self._base_dir, self._global_dict,
                                       phase_dict, self._failfast)
        elif "model" == phase_key:

            return Model.get_instance(self, self._base_dir, self._global_dict,
                                      phase_dict, self._failfast)
        elif "leapfrog" == phase_key:

            return Leapfrog.get_instance(self, self._base_dir,
                                         self._global_dict, phase_dict,
                                         self._failfast)
        elif "plotter" == phase_key:

            return Plot.get_instance(self, self._base_dir, self._global_dict,
                                     phase_dict, self._failfast)
        else:

            assert False
            return None

    def execute(self, *args, **kwargs):

        (args)
        (kwargs)

        results = []
        logger.info("Running config, name=%s" % (self._phase_dict["name"]))

        for phase_dict in self._phase_dict["phases"]:

            assert len(phase_dict) == 1
            phase_key = next(iter(phase_dict))
            phase_dict = phase_dict[phase_key]

            logger.info("Running phase, type=%s name=%s" %
                        (phase_key, phase_dict["name"]))
            phase = self._get_phase(phase_key, phase_dict)

            if phase.can_execute():

                results.append(phase.execute())

        return results


class ConfigManager(PhaseManager):

    def __init__(self, yaml_config):

        # Make sure that this is a filepath.
        assert pathlib.Path(yaml_config).resolve().exists()

        super(ConfigManager, self).__init__(yaml_config)

    def _copy_config_file(self, args):

        try:

            # Copy the config file to the base_dir.
            shutil.copy(args.config_file, args.base_dir)
        except shutil.SameFileError:

            # The src and dest are the same.
            # No changes required.
            pass

    def _create_experiment_config_file(self, args):

        experiment_config = open("%s/experiment.config" % (args.base_dir), "w")

        properties = {

            "experiment": {
                "name": constants.EXPERIMENT_NAME,
                "id": constants.EXPERIMENT_ID,
                "hostname": constants.HOSTNAME
            },

            "git": {
                "sha": constants.GIT_SHA,
                "branch": constants.GIT_BRANCH,
                "is_dirty": constants.GIT_IS_DIRTY
            },
        }

        json_str = json.dumps(properties, indent=1)
        experiment_config.write(json_str)

    def _create_git_info(self, args):

        file_handle = open("%s/git.status" % (args.base_dir), "w")
        completed_process = subprocess.run(
            "git status",
            shell=True,
            stdout=file_handle)
        file_handle.close()
        completed_process.check_returncode()

        file_handle = open("%s/git.diff" % (args.base_dir), "w")
        completed_process = subprocess.run(
            "git diff origin/HEAD",
            shell=True,
            stdout=file_handle)
        file_handle.close()
        completed_process.check_returncode()

    def run(self, args):

        phase.initialize_directories(args.base_dir, args.clean)
        self._copy_config_file(args)
        self._create_experiment_config_file(args)
        self._create_git_info(args)

        results = []
        for config in self._phase_dict:

            config_dict = config["config"]
            config = Config(args.base_dir, self._global_dict, config_dict,
                            args.failfast)

            if config.can_execute():

                results.append(config.execute())

        return results


def main(args):

    # Set the experiment name.
    constants.set_experiment_name(args.experiment_name)

    # Setup the logger.
    FORMAT = '(%(asctime)s) [%(levelname)9s]: %(message)s'
    logging.basicConfig(format=FORMAT)
    logger.setLevel(logging.INFO)

    # Run the config!
    config_manager = ConfigManager(args.config_file)
    config_manager.run(args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Learn generalized heuristics")

    # General arguments.
    parser.add_argument("--base-dir", default="./config",
                        help="Set up the base dir for the experiment")
    parser.add_argument("--clean", default=False,
                        action="store_true", help="Clean an existing config")
    parser.add_argument("--failfast", default=False,
                        action="store_true", help="Fail on the first error")

    parser.add_argument("--config-file", help="The path to the config file")
    parser.add_argument("--experiment-name", default=None,
                        help="The experiment name.")

    args = parser.parse_args()

    total_time = datetime.datetime.now()
    main(args)
    total_time = datetime.datetime.now() - total_time

    logger.info("Total time elapsed: %s" % (total_time))
