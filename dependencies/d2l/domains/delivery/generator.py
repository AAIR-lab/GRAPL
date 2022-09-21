#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import os
import random

from tarski.fstrips import create_fstrips_problem, language
from tarski.io import FstripsWriter
from tarski.syntax import land
from tarski.theories import Theory
from tarski import fstrips as fs

_CURRENT_DIR_ = os.path.dirname(os.path.realpath(__file__))


def create_actions(problem, add_fuel):
    lang = problem.language
    at, cell_t, empty, carrying, adjacent = lang.get("at", "cell", "empty", "carrying", "adjacent")

    t = lang.variable("t", 'truck')
    p = lang.variable("p", 'package')
    x = lang.variable("x", cell_t)
    f = lang.variable("from", cell_t)
    to = lang.variable("to", cell_t)
    
    problem.action(name='pick-package', parameters=[t, p, x],
                   precondition=land(at(p, x), at(t, x), empty(t), flat=True),
                   effects=[
                       fs.DelEffect(at(p, x)),
                       fs.DelEffect(empty(t)),
                       fs.AddEffect(carrying(t, p)),
                   ])

    problem.action(name='drop-package', parameters=[t, p, x],
                   precondition=land(at(t, x), carrying(t, p), flat=True),
                   effects=[
                       fs.AddEffect(empty(t)),
                       fs.DelEffect(carrying(t, p)),
                       fs.AddEffect(at(p, x)),
                   ])

    problem.action(name='move', parameters=[t, f, to],
                   precondition=land(adjacent(f, to), at(t, f), flat=True),
                   effects=[
                       fs.DelEffect(at(t, f)),
                       fs.AddEffect(at(t, to)),
                   ])


def generate_domain(gridsize, npackages, add_fuel=True):
    lang = language(theories=[Theory.EQUALITY, Theory.ARITHMETIC])
    problem = create_fstrips_problem(domain_name='delivery',
                                     problem_name=f"delivery-{gridsize}x{gridsize}-{npackages}",
                                     language=lang)

    cell_t = lang.sort('cell')
    lang.sort('locatable')
    lang.sort('package', 'locatable')
    lang.sort('truck', 'locatable')

    at = lang.predicate('at', 'locatable', 'cell')
    lang.predicate('carrying', 'truck', 'package')
    empty = lang.predicate('empty', 'truck')

    adjacent = lang.predicate('adjacent', cell_t, cell_t)

    # Create the actions
    create_actions(problem, add_fuel)

    rng = range(0, gridsize)
    coordinates = list(itertools.product(rng, rng))

    def cell_name(x, y):
        return f"c_{x}_{y}"

    truck = lang.constant('t1', 'truck')

    # Declare the constants:
    coord_objects = [lang.constant(cell_name(x, y), cell_t) for x, y in coordinates]

    package_objects = [lang.constant(f"p{i}", "package") for i in range(1, npackages+1)]

    # Declare the adjacencies:
    adjacent_coords = [(a, b, c, d) for (a, b), (c, d) in itertools.combinations(coordinates, 2)
                       if abs(a-c) + abs(b-d) == 1]

    for a, b, c, d in adjacent_coords:
        problem.init.add(adjacent, cell_name(a, b), cell_name(c, d))
        problem.init.add(adjacent, cell_name(c, d), cell_name(a, b))

    cd = coord_objects[:]
    random.shuffle(cd)

    # Initial positions
    problem.init.add(at, truck, cd.pop())
    for p in package_objects:
        problem.init.add(at, p, cd.pop())

    problem.init.add(empty, truck)

    # Set the problem goal
    target = cd.pop()
    goal = [at(p, target) for p in package_objects]
    problem.goal = land(*goal, flat=True)

    if add_fuel:
        # Our approach is not yet too int-friendly :-(
        # fuel_level_t = lang.interval('fuel_level', lang.Integer, lower_bound=0, upper_bound=10)
        fuel_level_t = lang.sort('fuel_level')

        current_fuel = lang.function('current_fuel', fuel_level_t)
        loc_fuel = lang.function('loc_fuel', cell_t)
        max_fuel_level = lang.function('max_fuel_level', fuel_level_t)
        min_fuel_level = lang.function('min_fuel_level', fuel_level_t)

        # The whole succ-predicate stuff
        succ = lang.predicate("succ", fuel_level_t, fuel_level_t)
        levels = ["f{}".format(i) for i in range(0, 11)]
        _ = [lang.constant(c, fuel_level_t) for c in levels]  # Create the "integer" objects
        _ = [problem.init.add(succ, x, y) for x, y in zip(levels, levels[1:])]

        problem.init.set(current_fuel, random.choice(levels))
        problem.init.set(min_fuel_level, levels[0])
        problem.init.set(max_fuel_level, levels[-1])
        problem.init.set(loc_fuel, cd.pop())

    return problem


def main():

    for gridsize in [3, 4, 5, 7, 9]:
        for npacks in [2, 3]:
            for run in range(0, 3):
                problem = generate_domain(gridsize, npackages=npacks, add_fuel=False)
                writer = FstripsWriter(problem)
                writer.write(domain_filename=os.path.join(_CURRENT_DIR_, "domain.pddl"),  # We can overwrite the domain
                             instance_filename=os.path.join(_CURRENT_DIR_, f"instance_{gridsize}_{npacks}_{run}.pddl"),
                             domain_constants=[])


if __name__ == "__main__":
    main()
