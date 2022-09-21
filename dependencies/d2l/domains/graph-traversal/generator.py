#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import os
import random

from tarski.fstrips import create_fstrips_problem, language, DelEffect, AddEffect
from tarski.io import FstripsWriter
from tarski.theories import Theory
from tarski.syntax import forall, implies, exists, land, Tautology

_CURRENT_DIR_ = os.path.dirname(os.path.realpath(__file__))


def create_noop(problem):
    # A hackish no-op, to prevent the planner from detecting that the action is useless and pruning it
    lang = problem.language
    node_t, at = lang.get("node", "at")
    x = lang.variable("x", node_t)
    problem.action(name='noop', parameters=[x], precondition=at(x), effects=[AddEffect(at(x))])


def generate_propositional_domain(nnodes, nedges, add_noop=False):
    lang = language(theories=[Theory.EQUALITY])
    problem = create_fstrips_problem(domain_name='graph-traversal-strips',
                                     problem_name="graph-traversal-{}x{}".format(nnodes, nedges),
                                     language=lang)

    node_t = lang.sort('node')
    at = lang.predicate('at', node_t)
    adjacent = lang.predicate('adjacent', node_t, node_t)

    # Create the actions
    from_, to = [lang.variable(name, node_t) for name in ("from", "to")]
    problem.action(name='move', parameters=[from_, to],
                   precondition=adjacent(from_, to) & at(from_),
                   effects=[DelEffect(at(from_)),
                            AddEffect(at(to))])

    if add_noop:
        create_noop(problem)

    def node_name(i):
        return "n_{}".format(i)

    # Declare the constants:
    node_ids = list(range(0, nnodes))
    nodes = [lang.constant(node_name(i), node_t) for i in node_ids]

    # Declare the adjacencies:
    adjacencies = random.sample(list(itertools.permutations(nodes, 2)), nedges)
    for n1, n2 in adjacencies:
        problem.init.add(adjacent, n1, n2)

    source_node = nodes[0]
    target_node = nodes[-1]

    problem.init.add(at, source_node)
    problem.goal = at(target_node)

    return problem


def main():

    add_noop = True
    for nnodes in [5, 10, 15]:
        nedges = int(nnodes**2/2)
        problem = generate_propositional_domain(nnodes, nedges, add_noop)
        writer = FstripsWriter(problem)
        writer.write(domain_filename=os.path.join(_CURRENT_DIR_, "domain.pddl"),  # We can overwrite the domain
                     instance_filename=os.path.join(_CURRENT_DIR_, "instance_{}_{}.pddl".format(nnodes, nedges)))


if __name__ == "__main__":
    main()
