import pathlib
import sys

import jpype
import jpype.imports
from jpype.types import *


def update_pythonpath():

    root = (pathlib.Path(pathlib.Path(__file__).parent) / "../../")

    sys.path.append((root / "generalized_learning").as_posix())

    fd_root_path = root / "dependencies" / "fast-downward_stochastic"
    sys.path.append(fd_root_path.as_posix())

    d2l_root_path = root / "dependencies" / "d2l"
    sltp_root_path = d2l_root_path / "src"

    sys.path.append(sltp_root_path.as_posix())

    tarski_root_path = root / "dependencies" / "tarski"
    tarski_src_path = tarski_root_path / "src"

    sys.path.append(tarski_src_path.as_posix())


update_pythonpath()


# Enable Java imports
# Pull in types
# Launch the JVM
jpype.startJVM()

root = pathlib.Path(__file__).parent.as_posix()
jpype.addClassPath("%s/bin" % (root))
jpype.addClassPath("%s/lib/*" % (root))

# Import module
# print(System.getProperty("java.class.path"))
from generalized_learning.concretized.action import StochasticAction
from generalized_learning.concretized.state import State
from generalized_learning.util import file
from java.io import File
from java.lang import System
from pddl import conditions
from rddl import ActionGenerator
from rddl import RDDL
from rddl.parser import parser
from rddl.policy import RandomBoolPolicy
from rddl.sim import Simulator
from rddl.viz import GenericScreenDisplay
from util import constants


def _create_rddl_predicate(name, terms):

    args = ()
    for term in terms:

        args += (term._sConstValue, )

    return conditions.Atom(name, args)


def convert_rddl_predicate(rddl_predicate):

    return conditions.Atom(rddl_predicate._sPredName,
                           rddl_predicate._alTerms)


def convert_rddl_state(rddl_state, non_fluents=set()):

    state = set(non_fluents)

    for predicate_name in rddl_state._state:

        true_args = rddl_state._state[predicate_name]

        for true_arg_list in true_args:

            predicate = _create_rddl_predicate(predicate_name,
                                               true_arg_list)

            state.add(predicate)

    return State(state)


def get_rddl_action(rddl_action_list):

    if len(rddl_action_list) == 0:

        final_name = "(noop)"
        base_name = "noop"
        params = []
    else:

        assert len(rddl_action_list) == 1
        rddl_action = rddl_action_list[0]

        base_name = rddl_action._sPredName
        params = []
        final_name = "(%s" % (base_name)
        for param in rddl_action._alTerms:

            params.append(param._sConstValue)
            final_name += " %s" % (param._sConstValue)

        final_name += ")"

    stochastic_action = StochasticAction(final_name, params,
                                         base_name, float("inf"))

    # Change the action list so that direct application from outside an
    # RDDL problem instance will result in an exception.
    stochastic_action._actions = None

    return stochastic_action


def _get_rddl_pddl_param_str(params, types):

    string = ""
    for i in range(len(params)):

        param_name = "?p%u - %s" % (i, params[i]._STypeName)
        assert params[i]._STypeName in types

        string += " %s" % (param_name)

    return string.strip()


def _get_rddl_pddl_grounded_param_str(terms):

    string = ""
    for term in terms:

        string += " %s" % (term._sConstValue)

    return string.strip()


def write_rddl_problem_to_file(problem_filepath,
                               directory="/tmp"):

    rddl = RDDL()
    rddl.addOtherRDDL(
        parser.parse(File(problem_filepath)),
        "dummy_problem")

    # We only allow one domain.
    assert len(rddl._tmInstanceNodes.keys()) == 1

    rddl_problem_key = rddl._tmInstanceNodes.keys()[0]
    rddl_problem = rddl._tmInstanceNodes[rddl_problem_key]

    assert len(rddl_problem._hmObjects) == 0
    assert rddl_problem._nNonDefActions == 1
    assert rddl_problem._termCond is None

    domain_name = rddl_problem._sDomain
    problem_name = rddl_problem._sName

    file_handle = open("%s/%s.%s" % (directory,
                                     problem_name,
                                     constants.PROBLEM_FILE_EXT), "w")

    file_handle.write("(define (problem %s)\n" % (problem_name))
    file_handle.write("(:domain %s)\n" % (domain_name))

    non_fluents = rddl._tmNonFluentNodes[rddl_problem._sNonFluents]

    file_handle.write("(:objects")
    for type_key in non_fluents._hmObjects.keys():

        objs = non_fluents._hmObjects[type_key]._alObjects
        for obj in objs:

            file_handle.write("\n    %s - %s" % (obj._sConstValue, type_key))
    file_handle.write(")\n")

    file_handle.write("(:init")
    for rddl_predicate in rddl_problem._alInitState + non_fluents._alNonFluents:

        assert rddl_predicate._oValue == True

        file_handle.write("\n    (%s %s)" % (
            rddl_predicate._sPredName,
            _get_rddl_pddl_grounded_param_str(rddl_predicate._alTerms)))
    file_handle.write(")\n")

    file_handle.write("(:goal (and))")
    file_handle.write(")")

    file.write_properties(file_handle, {"rddl": True},
                          constants.PDDL_COMMENT_PREFIX)

    file_handle.close()


def write_rddl_domain_to_file(domain_filepath, directory="/tmp"):

    domain_name, types, predicates, actions = get_domain(domain_filepath)

    file_handle = open("%s/%s.%s" % (directory,
                                     domain_name,
                                     constants.DOMAIN_FILE_EXT), "w")

    file_handle.write("(define (domain %s)\n" % (domain_name))

    file_handle.write("(:requirements :typing)\n")

    file_handle.write("(:types")
    for obj_type in types:

        file_handle.write(" %s" % (obj_type))
    file_handle.write(")\n")

    file_handle.write("(:predicates")

    for name, params in predicates:

        predicate_str = "\n    (%s %s)" % (
            name,
            _get_rddl_pddl_param_str(params, types))
        file_handle.write("%s" % (predicate_str))

    file_handle.write(")\n")

    file_handle.write("""(:action noop
    :parameters ()
    :precondition (and)
    :effect (and))""")

    for name, params in actions:

        action_str = """(:action %s
    :parameters (%s)
    :precondition (and)
    :effect (and))""" % (name, _get_rddl_pddl_param_str(params, types))

        file_handle.write("\n%s" % (action_str))

    file_handle.write(")")

    file.write_properties(file_handle, {"rddl": True},
                          constants.PDDL_COMMENT_PREFIX)

    file_handle.close()


def get_domain(domain_filepath):

    rddl = RDDL()
    rddl.addOtherRDDL(
        parser.parse(File(domain_filepath)),
        "dummy_domain")

    # We only allow one domain.
    assert len(rddl._tmDomainNodes.keys()) == 1

    rddl_domain_key = rddl._tmDomainNodes.keys()[0]
    rddl_domain = rddl._tmDomainNodes[rddl_domain_key]

    domain_name = rddl_domain._sDomainName

    types = set()
    for rddl_type in rddl_domain._hmTypes.keys():

        types.add(rddl_type._STypeName)

    print(types)

    predicates = []
    actions = []
    for rddl_variable in rddl_domain._hmPVariables:

        name = rddl_variable
        data = rddl_domain._hmPVariables[rddl_variable]

        if data._typeRange._STypeName != "bool":

            continue

        assert data._pvarName == name
        params = data._alParamTypes

        print(name, type(rddl_domain._hmPVariables[rddl_variable]))
        if type(rddl_domain._hmPVariables[rddl_variable]) == RDDL.PVARIABLE_ACTION_DEF:

            actions.append((name, params))
        else:

            predicates.append((name, params))

    return domain_name, types, predicates, actions


if __name__ == "__main__":

    print("hi")

    rddl = RDDL()
    rddl.addOtherRDDL(
        parser.parse(
            File("%s/files/final_comp_2014/rddl/wildfire_mdp.rddl" % (root))),
        "wildfire_mdp")

    rddl.addOtherRDDL(
        parser.parse(
            File("%s/files/final_comp_2014/rddl/wildfire_inst_mdp__1.rddl" % (root))),
        "wildfire_inst_mdp__1",
    )

    domain_filepath = "%s/files/final_comp_2014/rddl/wildfire_mdp.rddl" % (
        root)
    domain_name = "wildfire_mdp"

    write_rddl_domain_to_file(domain_filepath)

    problem_filepath = "%s/files/final_comp_2014/rddl/wildfire_inst_mdp__1.rddl" % (
        root)
    write_rddl_problem_to_file(problem_filepath)

    sim = Simulator(rddl, "wildfire_inst_mdp__1")
    pol = RandomBoolPolicy("wildfire_inst_mdp__1")
    pol.setRandSeed(0xDEADC0DE)
    pol.setRDDL(rddl)

    viz = GenericScreenDisplay()

    sim.resetState()

    # sim.run(pol, viz, 0xDEADC0DE)
    actions = ActionGenerator.getLegalBoolActionMap(sim._state)
    print(actions)

    for action_key in actions:

        print(get_rddl_action(actions[action_key]))

    domain_filepath = pathlib.Path("/tmp/wildfire_mdp.domain.pddl")
    from generalized_learning.concretized.domain import Domain
    domain = Domain(domain_filepath)

    from generalized_learning.concretized import problem as problem_instance
    problem = problem_instance.create_problem("wildfire_mdp.domain.pddl",
                                              "wildfire_inst_mdp__1.problem.pddl",
                                              "/tmp")

    actions = problem.get_applicable_actions(None)
    next_state, reward, done = problem.apply_action(actions[0], None)

    pass
