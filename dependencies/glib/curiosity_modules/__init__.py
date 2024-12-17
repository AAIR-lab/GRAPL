from .curiosity_base import BaseCuriosityModule
from .oracle_curiosity import OracleCuriosityModule
from .random_actions import RandomCuriosityModule
from .GLIB_grounded import *
from .GLIB_lifted import *


def create_curiosity_module(curiosity_module_name, action_space,
                            observation_space, planning_module,
                            learned_operators, operator_learning_module, domain_name):
    module = None
    if curiosity_module_name == "oracle":
        module = OracleCuriosityModule
    elif curiosity_module_name == "random":
        module = RandomCuriosityModule
    elif curiosity_module_name == "GLIB_G1":
        module = GLIBG1CuriosityModule
    elif curiosity_module_name == "GLIB_L2":
        module = GLIBL2CuriosityModule
    else:
        raise Exception("Unrecognized curiosity module '{}'".format(
            curiosity_module_name))
    return module(action_space, observation_space, planning_module,
                  learned_operators, operator_learning_module, domain_name)
