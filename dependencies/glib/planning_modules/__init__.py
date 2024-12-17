from .ff import FastForwardPlanner
from .ffreplan import FFReplanner

def create_planning_module(planning_module_name, learned_operators, domain_name,
                           action_space, observation_space):
    if planning_module_name.lower() == "ff":
        return FastForwardPlanner(learned_operators, domain_name, action_space, 
            observation_space)
    if planning_module_name.lower() == "ffreplan":
        return FFReplanner(learned_operators, domain_name, action_space, 
            observation_space)
    raise Exception("Unrecognized planning module '{}'".format(planning_module_name))
