#!/usr/bin/env python3

from abc import ABC
from abc import abstractmethod
from abc import abstractstaticmethod
import copy
import logging
import pathlib
import shutil

import yaml


logger = logging.getLogger(__name__)


def initialize_directories(base_dir, clean):

    base_dir = pathlib.Path(base_dir)
    logger.info("Setting up base_dir=%s" % (base_dir))

    # First, try to create the directory.
    #
    # This command will raise an exception if the path exists but is not a
    # directory and will be a NOP otherwise (ie if the directory already
    # exists then this won't do anything).
    base_dir.mkdir(mode=Phase.DIRECTORY_MODE, parents=True, exist_ok=True)

    # Now, if we need to clean the directory, go ahead and do it now.
    # Create a new directory once we cleant the tree.
    #
    # WARNING
    # ========
    #    The directory could have the potential to be cleant and recreated
    #    by the same run. We do not check against that.
    if clean:

        try:

            shutil.rmtree(base_dir)
            logger.info("Cleant base_dir=%s" % (base_dir))
        except FileNotFoundError:

            # Allow exceptions where the file never existed.
            logger.warning(
                "Could not clean base_dir=%s since it does not exist"
                % (base_dir))

        base_dir.mkdir(mode=Phase.DIRECTORY_MODE, parents=True,
                       exist_ok=True)

    return base_dir


class Phase(ABC):

    #: The set of required keys for this phase in order to function correctly.
    #:
    #: Use this so that we can detect errors at initialization rather than
    #: relying on the default value dict to throw an error later.
    REQUIRED_KEYS = set(["ignore", "name", "clean"])

    #: The default phase dict for this phase.
    DEFAULT_PHASE_DICT = {

        "base_dir": None,
        "ignore": False,
        "clean": False
    }

    #: The default directory mode to use when creating directories.
    DIRECTORY_MODE = 0o755

    @classmethod
    def _get_required_keys(cls):

        # Get the required keys of the subclass.
        #
        # NOTE
        # =====
        #    This only works for one level of inheritance.
        return copy.deepcopy(cls.REQUIRED_KEYS)

    @classmethod
    def _get_default_phase_dict(cls):

        return copy.deepcopy(cls.DEFAULT_PHASE_DICT)

    @staticmethod
    @abstractstaticmethod
    def get_instance(parent, parent_dir, global_dict, user_phase_dict,
                     failfast):

        raise NotImplementedError

    @abstractmethod
    def __init__(self, parent, parent_dir, global_dict={}, user_phase_dict={},
                 failfast=False):

        self._parent = parent
        self._parent_dir = parent_dir
        self._failfast = failfast

        # Just store these fields for debug.
        self._global_dict = global_dict
        self._user_phase_dict = user_phase_dict

        # Get the default phase dict, and override it with the global and user
        # provided phase dict's in order.
        #
        # Thus, the last override is always the user's provided values.
        self._phase_dict = self.__class__._get_default_phase_dict()
        self._override_dict(self._phase_dict, self._global_dict)
        self._override_dict(self._phase_dict, user_phase_dict)

        # Make sure that the required keys are present.
        assert self.__class__._get_required_keys() \
            .issubset(self._phase_dict.keys())

        # Initialize the base directory.
        self._base_dir = self._get_base_dir()

    def _get_base_dir(self):

        # Default base_dir is just parent_dir/name
        parent_dir = pathlib.Path(self._parent_dir)
        base_dir = parent_dir / self._phase_dict["name"]

        try:

            # Try to see if the user has provided a base_dir.
            base_dir = pathlib.Path(self._phase_dict["base_dir"])
            base_dir = base_dir.expanduser()
            base_dir = pathlib.Path(base_dir)

            if base_dir.is_absolute():

                logger.debug("base_dir=%s is an absolute path" % (base_dir))
            else:

                base_dir = parent_dir / base_dir
        except TypeError:

            pass

        # Now check if the base_dir is an absolute path.

        return base_dir

    def initialize_directories(self):

        return initialize_directories(self._base_dir,
                                      self._phase_dict["clean"])

    def _override_dict(self, phase_dict, override_dict):

        for key in override_dict:

            assert isinstance(key, str)

            if key not in phase_dict:

                if self._failfast:
                    raise Exception(
                        "%s cannot be overriden since it "
                        "is not in the phase_dict" % (key))
                else:

                    logger.debug(
                        "key=%s is being overriden even though it "
                        "does not exist in the phase_dict" % (key))

            # Override the value for the key.
            phase_dict[key] = override_dict[key]
            logger.debug("Override key=%s with value=%s"
                         % (key, str(phase_dict[key])))

    def can_execute(self):

        if self._phase_dict["ignore"]:

            logger.debug("Ignoring phase name=%s" % (self._phase_dict["name"]))
            return False
        else:

            return True

    @abstractmethod
    def execute(self, *args, **kwargs):

        (args)
        (kwargs)

        raise NotImplementedError

    def get_value(self, key):

        return self._phase_dict[key]

    def has_key(self, key):

        return key in self._phase_dict


class PhaseManager(ABC):

    @abstractmethod
    def __init__(self, yaml_config):

        # If the supplied parameter is a string, then it must mean that it is
        # a filepath.
        if isinstance(yaml_config, str):

            yaml_config = open(yaml_config, "r")

        logger.info("Loading YAML configuration from %s" % (str(yaml_config)))
        yaml_dict = yaml.load(yaml_config)

        try:
            self._global_dict = yaml_dict["globals"]
        except KeyError:

            raise Exception(
                "config YAML files must contain a 'globals' section")

        try:

            self._phase_dict = yaml_dict["phases"]
        except KeyError:

            raise Exception(
                "config YAML files must contain a 'phases' section")

    @abstractmethod
    def run(self, *args, **kwargs):

        raise NotImplementedError
