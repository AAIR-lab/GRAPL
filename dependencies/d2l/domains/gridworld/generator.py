#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from tarski.fstrips import create_fstrips_problem, language
from tarski.io import FstripsWriter
from tarski.theories import Theory
from tarski.syntax import land

_CURRENT_DIR_ = os.path.dirname(os.path.realpath(__file__))


def generate_domain(gridsize):
    lang = language(theories=[Theory.EQUALITY, Theory.ARITHMETIC])
    problem = create_fstrips_problem(domain_name='gridworld',
                                     problem_name="gridworld-{}x{}".format(gridsize, gridsize),
                                     language=lang)

    coord_t = lang.interval('coordinate', lang.Integer, lower_bound=1, upper_bound=gridsize)
    xpos = lang.function('xpos', coord_t)
    ypos = lang.function('ypos', coord_t)
    goal_xpos = lang.function('goal_xpos', coord_t)
    goal_ypos = lang.function('goal_ypos', coord_t)
    maxx = lang.function('maxpos', coord_t)

    # Create the actions
    problem.action(name='move-up', parameters=[],
                   precondition=ypos() < maxx(),
                   effects=[ypos() << ypos() + 1])

    problem.action(name='move-right', parameters=[],
                   precondition=xpos() < maxx(),
                   effects=[xpos() << xpos() + 1])

    problem.action(name='move-down', parameters=[],
                   precondition=ypos() > coord_t.lower_bound,
                   effects=[ypos() << ypos() - 1])

    problem.action(name='move-left', parameters=[],
                   precondition=xpos() > coord_t.lower_bound,
                   effects=[xpos() << xpos() - 1])

    problem.init.set(xpos, 1)
    problem.init.set(ypos, 1)
    problem.init.set(maxx, gridsize)

    problem.init.set(goal_xpos, gridsize)
    problem.init.set(goal_ypos, gridsize)

    problem.goal = land(xpos() == goal_xpos(), ypos() == goal_ypos())

    return problem


def generate_propositional_domain(gridsize):
    lang = language(theories=[Theory.EQUALITY])
    problem = create_fstrips_problem(domain_name='gridworld-strips',
                                     problem_name="gridworld-{}x{}".format(gridsize, gridsize),
                                     language=lang)

    coord_t = lang.sort('coordinate')
    xpos = lang.function('xpos', coord_t)
    ypos = lang.function('ypos', coord_t)
    maxx = lang.function('maxpos', coord_t)
    goal_xpos = lang.function('goal_xpos', coord_t)
    goal_ypos = lang.function('goal_ypos', coord_t)
    succ = lang.predicate("succ", coord_t, coord_t)

    coordinates = ["c{}".format(i) for i in range(1, gridsize+1)]
    _ = [lang.constant(c, coord_t) for c in coordinates]  # Create the "integer" objects

    x1 = lang.variable("x1", coord_t)

    # Create the actions
    problem.action(name='move_x', parameters=[x1],
                   precondition=succ(xpos(), x1) | succ(x1, xpos()),
                   effects=[xpos() << x1])

    problem.action(name='move_y', parameters=[x1],
                   precondition=succ(ypos(), x1) | succ(x1, ypos()),
                   effects=[ypos() << x1])

    last = coordinates[-1]
    problem.init.set(xpos, coordinates[0])
    problem.init.set(ypos, coordinates[0])
    problem.init.set(maxx, last)
    problem.init.set(goal_xpos, last)
    problem.init.set(goal_ypos, last)

    _ = [problem.init.add(succ, x, y) for x, y in zip(coordinates, coordinates[1:])]

    problem.goal = land(xpos() == goal_xpos(), ypos() == goal_ypos())

    return problem


def main():

    for gridsize in [3, 5, 7, 10]:
        problem = generate_domain(gridsize)
        writer = FstripsWriter(problem)
        writer.write(domain_filename=os.path.join(_CURRENT_DIR_, "domain.pddl"),  # We can overwrite the domain
                     instance_filename=os.path.join(_CURRENT_DIR_, "instance_{}.pddl".format(gridsize)))

        problem = generate_propositional_domain(gridsize)
        writer = FstripsWriter(problem)
        writer.write(domain_filename=os.path.join(_CURRENT_DIR_, "domain_strips.pddl"),  # We can overwrite the domain
                     instance_filename=os.path.join(_CURRENT_DIR_, "instance_strips_{}.pddl".format(gridsize)))


if __name__ == "__main__":
    main()
