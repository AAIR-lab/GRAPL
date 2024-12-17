from ndr.ndrs import *
from ndr.learn import *
from pddlgym.structs import Type, Anti
from ndr.main import *
from ndr.utils import nostdout
import gym
import pddlgym
import pybullet_abstraction_envs
import numpy as np


VERBOSE = True

# Some shared stuff
block_type = Type("block")
Act0 = Predicate("act0" , 0, [])
Act01 = Predicate("act01" , 0, [])
Act1 = Predicate("act1" , 0, [block_type])
Red = Predicate("red", 1, [block_type])
Blue = Predicate("blue", 1, [block_type])
HandsFree0 = Predicate("HandsFree0", 0, [])

MoveableType = Type('moveable')
StaticType = Type('static')
IsRobot = Predicate('IsRobot', 1, var_types=[MoveableType])
IsBear = Predicate('IsBear', 1, var_types=[MoveableType])
IsHoney = Predicate('IsHoney', 1, var_types=[MoveableType])
IsPawn = Predicate('IsPawn', 1, var_types=[MoveableType])
IsMonkey = Predicate('IsMonkey', 1, var_types=[MoveableType])
IsGoal = Predicate('IsGoal', 1, var_types=[StaticType])
At = Predicate('At', 2, var_types=[MoveableType, StaticType])
Holding = Predicate('Holding', 1, var_types=[MoveableType])
HandsFree = Predicate("HandsFree", 1, var_types=[MoveableType])
MoveTo = Predicate('MoveTo', 1, var_types=[StaticType])
Pick = Predicate('Pick', 1, var_types=[MoveableType])
Place = Predicate('Place', 1, var_types=[MoveableType])
Pet = Predicate('Pet', 1, var_types=[MoveableType])
PutOn = Predicate('PutOn', 1, var_types=[MoveableType])
WantHolding = Predicate('WantHolding', 1, var_types=[MoveableType])
WantAt = Predicate('WantAt', 2, var_types=[MoveableType, StaticType])
On = Predicate('On', 2, var_types=[MoveableType, MoveableType])

PlaceType = Type('place')
PathType = Type('path')
In = Predicate('in', 1, var_types=[PlaceType])
Visited = Predicate('visited', 1, var_types=[PlaceType])
Notvisited = Predicate('not-visited', 1, var_types=[PlaceType])
Complete = Predicate('complete', 1, var_types=[PathType])
Notcomplete = Predicate('not-complete', 1, var_types=[PathType])
Connected = Predicate('connected', 2, var_types=[PlaceType, PlaceType])
Start = Predicate('start', 1, var_types=[PlaceType])
TSPMoveTo = Predicate('moveto', 1, var_types=[PlaceType])


def test_ndr():
    def create_ndr():
        action = Act0()
        preconditions = [Red("?x"), HandsFree0()]
        effect_probs = [0.8, 0.2]
        effects = [{Anti(HandsFree0())}, {NOISE_OUTCOME}]
        return NDR(action, preconditions, effect_probs, effects)

    # Test copy
    ndr = create_ndr()
    ndr_copy = ndr.copy()
    ndr.preconditions.remove(HandsFree0())
    assert HandsFree0() not in ndr.preconditions
    assert HandsFree0() in ndr_copy.preconditions
    del ndr.effects[0]
    assert len(ndr.effects) == 1
    assert len(ndr_copy.effects) == 2
    ndr.effect_probs[0] = 1.0
    assert ndr.effect_probs[0] == 1.0
    assert ndr_copy.effect_probs[0] == 0.8

    # Test find substitutions
    ndr = create_ndr()
    state = {Red("block0")}
    action = Act0()
    assert ndr.find_substitutions(state, action) == None
    state = {Red("block0"), HandsFree0(), Blue("block1")}
    action = Act0()
    sigma = ndr.find_substitutions(state, action)
    assert sigma is not None
    assert len(sigma) == 1
    assert sigma[block_type("?x")] == block_type("block0")
    state = {Red("block0"), HandsFree0(), Red("block1")}
    action = Act0()
    assert ndr.find_substitutions(state, action) == None

    # Test find_unique_matching_effect_index
    ndr = create_ndr()
    state = {Red("block0"), HandsFree0(), Blue("block1")}
    action = Act0()
    effects = {Anti(HandsFree0())}
    assert ndr.find_unique_matching_effect_index((state, action, effects)) == 0
    state = {Red("block0"), HandsFree0(), Blue("block1")}
    action = Act0()
    effects = {Anti(HandsFree0()), Blue("block0")}
    assert ndr.find_unique_matching_effect_index((state, action, effects)) == 1

    print("Test NDR passed.")

def test_ndr_set():
    def create_ndr_set():
        action = Act0()
        preconditions = [Red("?x"), HandsFree0()]
        effect_probs = [0.8, 0.2]
        effects = [{Anti(HandsFree0())}, {NOISE_OUTCOME}]
        ndr0 = NDR(action, preconditions, effect_probs, effects)
        preconditions = [Red("?x"), Blue("?x")]
        effect_probs = [0.5, 0.4, 0.1]
        effects = [{HandsFree0()}, {Anti(Blue("?x"))}, {NOISE_OUTCOME}]
        ndr1 = NDR(action, preconditions, effect_probs, effects)
        return NDRSet(action, [ndr0, ndr1])

    # Test find rule
    ndr_set = create_ndr_set()
    state = {Red("block0")}
    action = Act0()
    assert ndr_set.find_rule((state, action, set())) == ndr_set.default_ndr
    state = {Red("block0"), HandsFree0(), Blue("block1")}
    action = Act0()
    assert ndr_set.find_rule((state, action, set())) == ndr_set.ndrs[0]
    state = {Red("block0"), Blue("block0")}
    action = Act0()
    assert ndr_set.find_rule((state, action, set())) == ndr_set.ndrs[1]

    # Test partition transitions
    transitions = [
        ({Red("block0"), HandsFree0(), Blue("block1")}, Act0(), set()),
        ({Red("block0"), Blue("block0")}, Act0(), set()),
        ({Red("block0"), Blue("block0"), Blue("block1")}, Act0(), set()),
        ({Red("block0")}, Act0(), set()),
    ]
    partitions = ndr_set.partition_transitions(transitions)
    assert partitions[0] == [transitions[0]]
    assert partitions[1] == transitions[1:3]
    assert partitions[2] == [transitions[3]]
    print("Test NDRSet passed.")

def test_planning():
    print("Running planning test with ground-truth NDRs")
    with nostdout():
        env = NDRBlocksEnv()
        env.seed(0)
        policy = find_policy("ff_replan", env.operators, env.action_space, env.observation_space)
        total_returns = 0
        for trial in range(10):
            returns = run_policy(env, policy, verbose=False, render=False, check_reward=False)
            total_returns += returns
    # print("Average returns:", total_returns/10.)
    assert total_returns == 6


def run_integration_test(training_data, test_transitions, expect_deterministic=True):
    """Assumes deterministic
    """
    rule_set = learn_rule_set(training_data)
    if VERBOSE:
        print("Learned rule set:")
        print_rule_set(rule_set)
    
    # Make sure rules are deterministic
    if expect_deterministic:
        for rules in rule_set.values():
            for rule in rules.ndrs:
                for p in rule.effect_probs:
                    assert abs(p) < 1e-6 or abs(1-p) < 1e-6

    # Test predictions
    for s, a, effs in test_transitions:
        action_rule_set = rule_set[a.predicate]
        prediction = action_rule_set.predict_max(s, a)
        if not sorted(effs) == sorted(prediction):
            import ipdb; ipdb.set_trace()

    print("Test passed")

def test_integration1():
    print("Running integration test 1...")

    training_data = {
        Place : [
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o1')},
             Place('o1'),
             {Anti(Holding('o1'))},
            ),
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o2')},
             Place('o1'),
             set(),
            ),
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o2')},
             Place('o2'),
             {Anti(Holding('o2'))},
            ),
        ]
    }

    test_transitions = [
        ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o3')},
         Place('o3'),
         {Anti(Holding('o3'))},
        ),
        ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o1')},
         Place('o3'),
         set(),
        ),
        ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o3')},
         Place('o1'),
         set(),
        ),
    ]

    return run_integration_test(training_data, test_transitions)

def test_integration2():
    """It was important to add TrimObjects for this test to pass
    """
    print("Running integration test 2...")

    training_data = {
        MoveTo : [
            ({At('robot', 'loc1'), At('o1', 'loc2'),
              IsMonkey('m1'), IsRobot('robot'), IsPawn('o1') },
             MoveTo('loc2'),
             {Anti(At('robot', 'loc1')), At('robot', 'loc2')},
            ),
            ({At('robot', 'loc1'), At('o1', 'loc2'), At('m1', 'loc1'), 
              IsMonkey('m1'), IsRobot('robot'), IsPawn('o1') },
             MoveTo('loc1'),
             set(),
            ),
            ({At('robot', 'loc1'), At('o1', 'loc2'), At('m1', 'loc1'),
              At('o1', 'loc3'), IsMonkey('m1'), IsRobot('robot'), IsPawn('o1') },
             MoveTo('loc3'),
             {Anti(At('robot', 'loc1')), At('robot', 'loc3')},
            ),
            ({At('robot', 'loc1'), At('o1', 'loc2'), At('m1', 'loc1'),
              At('o1', 'loc3'), IsMonkey('m1'), IsRobot('robot'), IsMonkey('o1') },
             MoveTo('loc3'),
             {Anti(At('robot', 'loc1')), At('robot', 'loc3')},
            ),
            ({At('robot', 'loc1'), At('o1', 'loc2'), At('m1', 'loc1'),
              At('o1', 'loc3'), IsPawn('m1'), IsRobot('robot'), IsPawn('o1') },
             MoveTo('loc3'),
             {Anti(At('robot', 'loc1')), At('robot', 'loc3')},
            ),
        ]
    }

    test_transitions = [
        ({At('robot', 'loc1'), At('o1', 'loc4'), IsRobot('robot')},
         MoveTo('loc4'),
         {Anti(At('robot', 'loc1')), At('robot', 'loc4')},
        ),
        ({At('robot', 'loc1'), At('o1', 'loc4')},
         MoveTo('loc4'),
         set(),
        ),
        ({At('m1', 'loc1'), At('o1', 'loc4'), IsMonkey('m1')},
         MoveTo('loc4'),
         set(),
        ),
    ]

    return run_integration_test(training_data, test_transitions)

def test_integration3():
    print("Running integration test 3...")

    training_data = {
        Pick : [
            ({ IsPawn('o1'), IsPawn('o2'), IsRobot('robot'), At('o1', 'loc1'), 
              At('o2', 'loc2'), At('robot', 'loc1'), HandsFree0(), },
             Pick('o1'),
             { Holding('o1'), Anti(At('o1', 'loc1')), Anti(HandsFree0()) }),
            ({ IsPawn('o1'), IsPawn('o2'), IsRobot('robot'), At('o1', 'loc1'), 
              At('o2', 'loc2'), At('robot', 'loc1'), HandsFree0(), },
             Pick('o2'),
             set()),
            ({ IsPawn('o1'), IsPawn('o2'), IsRobot('robot'), At('o1', 'loc1'), 
              At('o2', 'loc2'), At('robot', 'loc2'), HandsFree0(), },
             Pick('o2'),
             { Holding('o2'), Anti(At('o2', 'loc2')), Anti(HandsFree0()) }),
            ({ IsPawn('o1'), IsPawn('o2'), IsRobot('robot'), At('o1', 'loc2'), 
              At('o2', 'loc2'), At('robot', 'loc2'), HandsFree0(), },
             Pick('o2'),
             { Holding('o2'), Anti(At('o2', 'loc2')), Anti(HandsFree0()) }),
            ({ IsPawn('o1'), IsPawn('o2'), IsRobot('robot'), At('o1', 'loc1'), 
              At('o2', 'loc2'), At('robot', 'loc2'), HandsFree0(), },
             Pick('o1'),
             set()),
            ({ IsPawn('o1'), IsPawn('o2'), IsRobot('robot'), At('o1', 'loc1'), 
              At('o2', 'loc2'), At('robot', 'loc2'), Holding('o3'), },
             Pick('o2'),
             set()),
        ],
    }

    test_transitions = [
        ({ IsPawn('o1'), IsPawn('o2'), IsRobot('robot'), At('o1', 'loc1'), 
          At('o2', 'loc1'), At('robot', 'loc1'), HandsFree0(), },
         Pick('o1'),
         { Holding('o1'), Anti(At('o1', 'loc1')), Anti(HandsFree0()) }),
        ({ IsPawn('o1'), IsPawn('o2'), At('o1', 'loc1'), 
          At('o2', 'loc1'), At('robot', 'loc1'), HandsFree0(), },
         Pick('o1'),
         set()),
    ]

    return run_integration_test(training_data, test_transitions)

def test_integration4():
    print("Running integration test 4...")

    training_data = {
        Pick : [
            ({ IsPawn('o1'), IsPawn('o2'), IsRobot('robot'), At('o1', 'loc1'), 
              At('o2', 'loc2'), At('robot', 'loc1'), HandsFree('robot'), },
             Pick('o1'),
             { Holding('o1'), Anti(At('o1', 'loc1')), Anti(HandsFree('robot')) }),
            ({ IsPawn('o1'), IsPawn('o2'), IsRobot('robot'), At('o1', 'loc1'), 
              At('o2', 'loc2'), At('robot', 'loc1'), HandsFree('robot'), },
             Pick('o2'),
             set()),
            ({ IsPawn('o1'), IsPawn('o2'), IsRobot('robot'), At('o1', 'loc1'), 
              At('o2', 'loc2'), At('robot', 'loc2'), HandsFree('robot'), },
             Pick('o2'),
             { Holding('o2'), Anti(At('o2', 'loc2')), Anti(HandsFree('robot')) }),
            ({ IsPawn('o1'), IsPawn('o2'), IsRobot('robot'), At('o1', 'loc1'), 
              At('o2', 'loc2'), At('robot', 'loc2'), HandsFree('robot'), },
             Pick('o1'),
             set()),
            ({ IsPawn('o1'), IsPawn('o2'), IsRobot('robot'), At('o1', 'loc1'), 
              At('o2', 'loc2'), At('robot', 'loc2'), HandsFree('robot'), },
             Pick('robot'),
             set()),
            # Identical to above, but with monkeys instead of pawns
            ({ IsMonkey('o1'), IsMonkey('o2'), IsRobot('robot'), At('o1', 'loc1'), 
              At('o2', 'loc2'), At('robot', 'loc1'), HandsFree('robot'), },
             Pick('o1'),
             { Holding('o1'), Anti(At('o1', 'loc1')), Anti(HandsFree('robot')) }),
            ({ IsMonkey('o1'), IsMonkey('o2'), IsRobot('robot'), At('o1', 'loc1'), 
              At('o2', 'loc2'), At('robot', 'loc1'), HandsFree('robot'), },
             Pick('o2'),
             set()),
            ({ IsMonkey('o1'), IsMonkey('o2'), IsRobot('robot'), At('o1', 'loc1'), 
              At('o2', 'loc2'), At('robot', 'loc2'), HandsFree('robot'), },
             Pick('o2'),
             { Holding('o2'), Anti(At('o2', 'loc2')), Anti(HandsFree('robot')) }),
            ({ IsMonkey('o1'), IsMonkey('o2'), IsRobot('robot'), At('o1', 'loc1'), 
              At('o2', 'loc2'), At('robot', 'loc2'), HandsFree('robot'), },
             Pick('o1'),
             set()),
            ({ IsMonkey('o1'), IsMonkey('o2'), IsRobot('robot'), At('o1', 'loc1'), 
              At('o2', 'loc2'), At('robot', 'loc2'), HandsFree('robot'), },
             Pick('robot'),
             set()),
        ],
    }

    test_transitions = [
        ({ IsPawn('o1'), IsPawn('o2'), IsRobot('robot'), At('o1', 'loc1'), 
          At('o2', 'loc1'), At('robot', 'loc1'), HandsFree('robot'), },
         Pick('o1'),
         { Holding('o1'), Anti(At('o1', 'loc1')), Anti(HandsFree('robot')) }),
        ({ IsPawn('o1'), IsPawn('o2'), At('o1', 'loc1'), 
          At('o2', 'loc2'), At('robot', 'loc1'), HandsFree('robot'), },
         Pick('o2'),
         set()),
        ({ IsPawn('o1'), IsPawn('o2'), IsRobot('robot'), At('o1', 'loc1'), 
          At('o2', 'loc1'), At('robot', 'loc1'), HandsFree('robot'), },
         Pick('robot'),
         set()),
    ]

    return run_integration_test(training_data, test_transitions)

def test_integration5():
    print("Running integration test 5...")

    training_data = {
        TSPMoveTo : [
            ({ Connected('a', 'b'), Connected('b', 'c'), Connected('c', 'd'),
               Connected('c', 'b'), Connected('d', 'a'), Start('a'),
               In('a'),
               Visited('a'), Notvisited('b'), Notvisited('c'), Notvisited('d'),
               Notcomplete('path'),},
             TSPMoveTo('c'),
             set()),
            ({ Connected('a', 'b'), Connected('b', 'c'), Connected('c', 'd'),
               Connected('c', 'b'), Connected('d', 'a'), Start('a'),
               In('a'),
               Visited('a'), Notvisited('b'), Notvisited('c'), Notvisited('d'),
               Notcomplete('path'),},
             TSPMoveTo('b'),
             { Visited('b'), In('b'), Anti(In('a')), Anti(Notvisited('b'))}),
            ({ Connected('a', 'b'), Connected('b', 'c'), Connected('c', 'd'),
               Connected('c', 'b'), Connected('d', 'a'), Start('a'),
               In('b'),
               Visited('a'), Visited('b'), Notvisited('c'), Notvisited('d'),
               Notcomplete('path'),},
             TSPMoveTo('a'),
             set()),
            ({ Connected('a', 'b'), Connected('b', 'c'), Connected('c', 'd'),
               Connected('c', 'b'), Connected('d', 'a'), Start('a'),
               In('b'),
               Visited('a'), Visited('b'), Notvisited('c'), Notvisited('d'),
               Notcomplete('path'),},
             TSPMoveTo('c'),
             { Visited('c'), In('c'), Anti(In('b')), Anti(Notvisited('c'))}),
            ({ Connected('a', 'b'), Connected('b', 'c'), Connected('c', 'd'),
               Connected('c', 'b'), Connected('d', 'a'), Start('a'),
               In('b'),
               Visited('a'), Visited('b'), Visited('c'), Notvisited('d'),
               Notcomplete('path'),},
             TSPMoveTo('b'),
             set()),
            # this is a crucial example!
            ({ Connected('a', 'b'), Connected('b', 'c'), Connected('c', 'd'),
               Connected('c', 'b'), Connected('d', 'a'), Start('a'),
               In('c'),
               Visited('a'), Visited('b'), Visited('c'), Notvisited('d'),
               Notcomplete('path'),},
             TSPMoveTo('b'),
             set()),
            ({ Connected('a', 'b'), Connected('b', 'c'), Connected('c', 'd'),
               Connected('c', 'b'), Connected('d', 'a'), Start('a'),
               In('d'),
               Visited('a'), Visited('b'), Visited('c'), Visited('d'),
               Notcomplete('path'),},
             TSPMoveTo('a'),
             { In('a'), Anti(In('d')), Anti(Notcomplete('path')), Complete('path')}),
            # different graph
            ({ Connected('a', 'b'), Connected('b', 'c'), Connected('c', 'd'), Connected('d', 'c'),
               Connected('c', 'e'), Connected('e', 'f'), Connected('f', 'a'), Start('a'),
               In('a'),
               Visited('a'), Notvisited('b'), Notvisited('c'), Notvisited('d'), 
               Notvisited('e'), Notvisited('f'), Notcomplete('path'),},
             TSPMoveTo('d'),
             set()),
            ({ Connected('a', 'b'), Connected('b', 'c'), Connected('c', 'd'), Connected('d', 'c'),
               Connected('c', 'e'), Connected('e', 'f'), Connected('f', 'a'), Start('a'),
               In('a'),
               Visited('a'), Notvisited('b'), Notvisited('c'), Notvisited('d'), 
               Notvisited('e'), Notvisited('f'), Notcomplete('path'),},
             TSPMoveTo('b'),
             { Visited('b'), In('b'), Anti(In('a')), Anti(Notvisited('b'))}),
            ({ Connected('a', 'b'), Connected('b', 'c'), Connected('c', 'd'), Connected('d', 'c'),
               Connected('c', 'e'), Connected('e', 'f'), Connected('f', 'a'), Start('a'),
               In('c'),
               Visited('a'), Visited('b'), Visited('c'), Notvisited('d'), 
               Notvisited('e'), Notvisited('f'), Notcomplete('path'),},
             TSPMoveTo('e'),
             { Visited('e'), In('e'), Anti(In('c')), Anti(Notvisited('e'))}),
            ({ Connected('a', 'b'), Connected('b', 'c'), Connected('c', 'd'), Connected('d', 'c'),
               Connected('c', 'e'), Connected('e', 'f'), Connected('f', 'a'), Start('a'),
               In('f'),
               Visited('a'), Visited('b'), Visited('c'), Notvisited('d'), 
               Visited('e'), Visited('f'), Notcomplete('path'),},
             TSPMoveTo('d'),
             set()),
            # crucial example
            ({ Connected('a', 'b'), Connected('b', 'c'), Connected('c', 'd'), Connected('d', 'c'),
               Connected('c', 'e'), Connected('e', 'f'), Connected('f', 'a'), Start('a'),
               In('d'),
               Visited('a'), Visited('b'), Visited('c'), Visited('d'), 
               Notvisited('e'), Notvisited('f'), Notcomplete('path'),},
             TSPMoveTo('c'),
             set()),
            ({ Connected('a', 'b'), Connected('b', 'c'), Connected('c', 'd'), Connected('d', 'c'),
               Connected('c', 'e'), Connected('e', 'f'), Connected('f', 'a'), Start('a'),
               In('f'),
               Visited('a'), Visited('b'), Visited('c'), Notvisited('d'), 
               Visited('e'), Visited('f'), Notcomplete('path'),},
             TSPMoveTo('c'),
             set()),
            ({ Connected('a', 'b'), Connected('b', 'c'), Connected('c', 'd'), Connected('d', 'c'),
               Connected('c', 'e'), Connected('e', 'f'), Connected('f', 'a'), Start('a'),
               In('f'),
               Visited('a'), Visited('b'), Visited('c'), Notvisited('d'), 
               Visited('e'), Visited('f'), Notcomplete('path'),},
             TSPMoveTo('a'),
             { In('a'), Anti(In('f')), Anti(Notcomplete('path')), Complete('path')}),
            # different graph
            ({ Connected('a', 'b'), Start('a'),
               In('a'),
               Visited('a'), Notvisited('b'), Notcomplete('path'),},
             TSPMoveTo('a'),
             set()),
            ({ Connected('a', 'b'), Start('a'),
               In('a'),
               Visited('a'), Notvisited('b'), Notcomplete('path'),},
             TSPMoveTo('b'),
             { In('b'), Anti(In('a')), Visited('b'), Anti(Notvisited('b')) }),
        ],
    }

    # todo
    test_transitions = [
    ]

    return run_integration_test(training_data, test_transitions)

def test_integration6():
    # Minimal noise test
    print("Running integration test 6...")

    training_data = {
        Place : [
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o1')},
             Place('o1'),
             {Anti(Holding('o1'))},
            ),
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o2')},
             Place('o1'),
             set(),
            ),
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o2')},
             Place('o2'),
             {Anti(Holding('o2'))},
            ),
            # repeat
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o1')},
             Place('o1'),
             {Anti(Holding('o1'))},
            ),
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o2')},
             Place('o1'),
             set(),
            ),
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o2')},
             Place('o2'),
             {Anti(Holding('o2'))},
            ),
            # repeat
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o1')},
             Place('o1'),
             {Anti(Holding('o1'))},
            ),
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o2')},
             Place('o1'),
             set(),
            ),
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o2')},
             Place('o2'),
             {Anti(Holding('o2'))},
            ),
            # noise
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o1')},
             Place('o1'),
             set(),
            ),
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o1')},
             Place('o1'),
             {Anti(Holding('o1')), Anti(IsPawn('o2'))},
            ),
        ]
    }

    test_transitions = [
        ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o3')},
         Place('o3'),
         {Anti(Holding('o3'))},
        ),
        ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o1')},
         Place('o3'),
         set(),
        ),
        ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o3')},
         Place('o1'),
         set(),
        ),
    ]

    return run_integration_test(training_data, test_transitions, expect_deterministic=False)

def test_integration7():
    # Noisy puton test
    print("Running integration test 7...")

    training_data = {
        PutOn : [
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o1')},
             PutOn('o2'),
             {Anti(Holding('o1')), On('o1', 'o2')},
            ),
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o1')},
             PutOn('o3'),
             {Anti(Holding('o1')), On('o1', 'o3')},
            ),
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o1'), On('o2', 'o3')},
             PutOn('o2'),
             {Anti(Holding('o1')), On('o1', 'o2')},
            ),
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o1'), On('o2', 'o3')},
             PutOn('o3'),
             {Anti(Holding('o1')), On('o1', 'o2')},
            ),
            # this is an important one
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o1'), On('o2', 'o3')},
             PutOn('o3'),
             {Anti(Holding('o1'))},
            ),
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o1')},
             PutOn('o1'),
             set(),
            ),
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o1')},
             PutOn('o1'),
             {Anti(Holding('o1'))},
            ),
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o1')},
             PutOn('o1'),
             {Anti(Holding('o1')), On('o1', 'o3')},
            ),
            ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3')},
             PutOn('o2'),
             set(),
            ),
        ]
    }

    test_transitions = [
        ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o2')},
         PutOn('o1'),
         {Anti(Holding('o2')), On('o2', 'o1')},
        ),
        ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o2'), On('o1', 'o3')},
         PutOn('o1'),
         {Anti(Holding('o2')), On('o2', 'o1')},
        ),
        ({IsPawn('o1'), IsPawn('o2'), IsPawn('o3'), Holding('o1'), On('o2', 'o3')},
         PutOn('o3'),
         {Anti(Holding('o1'))},
        ),
    ]

    return run_integration_test(training_data, test_transitions, expect_deterministic=False)

def test_negative_preconditions():
    # This came up in the course of integration test 7

    # ?x0 must bind to o0 and ?x1 must bind to o1, so ?x2 must bind to o2
    conds = [ PutOn("?x0"), Holding("?x1"), IsPawn("?x2"), Not(On("?x2", "?x0")) ]
    kb = { PutOn('o0'), IsPawn('o0'), IsPawn('o1'), IsPawn('o2'), Holding('o1'), }
    assignments = find_satisfying_assignments(kb, conds, allow_redundant_variables=False)
    assert len(assignments) == 1

    # should be the same, even though IsPawn("?x2") is removed...
    conds = [ PutOn("?x0"), Holding("?x1"), Not(On("?x2", "?x0")) ]
    kb = { PutOn('o0'), IsPawn('o0'), IsPawn('o1'), IsPawn('o2'), Holding('o1'), }
    assignments = find_satisfying_assignments(kb, conds, allow_redundant_variables=False)
    assert len(assignments) == 1



def test_system():
    seed = 0
    print("Running end-to-end tests (this will take a long time)")

    # uncomment and indent below to suppress printouts
    # with nostdout():

    # Test Hanoi
    training_env = gym.make("PDDLEnvHanoi-v0")
    training_env.seed(seed)
    training_data = collect_training_data(training_env,
        num_transitions_per_problem=10,
        max_transitions_per_action=500)
    training_env.close()
    rule_set = learn_rule_set(training_data)
    test_env = gym.make("PDDLEnvHanoiTest-v0")
    test_results = run_test_suite(rule_set, test_env, render=False, verbose=False,
        num_problems=5,
        max_num_steps=10000)
    test_env.close()
    assert np.sum(test_results) == 5
    print("Hanoi integration test passed.")

    # Test Doors
    training_env = gym.make("PDDLEnvDoors-v0")
    training_env.seed(seed)
    training_data = collect_training_data(training_env,
        num_transitions_per_problem=20,
        max_transitions_per_action=1000)
    training_env.close()
    rule_set = learn_rule_set(training_data)
    test_env = gym.make("PDDLEnvDoorsTest-v0")
    test_results = run_test_suite(rule_set, test_env, render=False, verbose=False,
        num_problems=5,
        max_num_steps=10000)
    test_env.close()
    assert np.sum(test_results) == 5
    print("Doors integration test passed.")

    # Test Rearrangement
    training_env = gym.make("PDDLEnvRearrangement-v0")
    training_env.seed(seed)
    training_data = collect_training_data(training_env,
        num_transitions_per_problem=10,
        max_transitions_per_action=500)
    training_env.close()
    rule_set = learn_rule_set(training_data)
    test_env = gym.make("PDDLEnvRearrangement-v0")
    test_results = run_test_suite(rule_set, test_env, render=False, verbose=False,
        num_problems=5,
        max_num_steps=10000)
    test_env.close()
    assert np.sum(test_results) == 5
    print("Rearrangement integration test passed.")

    # Test deterministic blocks
    training_env = gym.make("PDDLEnvBlocks-v0")
    training_env.seed(seed)
    training_data = collect_training_data(training_env,
        max_num_trials=10000,
        num_transitions_per_problem=10,
        max_transitions_per_action=1000,)
    training_env.close()
    rule_set = learn_rule_set(training_data)
    test_env = gym.make("PDDLEnvBlocksTest-v0")
    test_results = run_test_suite(rule_set, test_env, render=False, verbose=False,
        num_problems=5,
        max_num_steps=50)
    test_env.close()
    if not np.sum(test_results) == 5:
        import ipdb; ipdb.set_trace()
    print("Blocks integration test passed.")

    # Test NDRBlocks
    training_env = NDRBlocksEnv()
    training_env.seed(seed)
    training_data = collect_training_data(training_env)
    training_env.close()
    rule_set = learn_rule_set(training_data)
    test_env = NDRBlocksEnv()
    test_results = run_test_suite(rule_set, test_env, render=False, verbose=False,
        num_problems=100)
    test_env.close()
    assert 40 < np.sum(test_results) < 60
    print("NDRBlocks integration test passed.")

    # Test TSP
    training_env = gym.make("PDDLEnvTsp-v0")
    training_env.seed(seed)
    training_data = collect_training_data(training_env,
        max_num_trials=5000,
        num_transitions_per_problem=100,
        max_transitions_per_action=2500,)
    training_env.close()
    rule_set = learn_rule_set(training_data)
    test_env = gym.make("PDDLEnvTspTest-v0")
    test_results = run_test_suite(rule_set, test_env, render=False, verbose=False,
        num_problems=5,
        max_num_steps=10000)
    test_env.close()
    assert np.sum(test_results) == 5
    print("TSP integration test passed.")

    # Test PybulletBlocksEnv
    training_env = gym.make("PDDLEnvPybulletBlocks-v0") #PybulletBlocksEnv(use_gui=False)
    training_env.seed(seed)
    training_data = collect_training_data(training_env,
        # Cache this because it takes a very long time to create the dataset
        "/tmp/pybullet_blocks_env_integration_test_dataset.pkl",
        max_num_trials=5000,
        num_transitions_per_problem=1,
        max_transitions_per_action=500,
        verbose=True)
    training_env.close()
    rule_set = learn_rule_set(training_data)
    print_rule_set(rule_set)
    test_env = gym.make("PDDLEnvPybulletBlocksTest-v0")  #PybulletBlocksEnv(use_gui=False)
    test_results = run_test_suite(rule_set, test_env, render=False, verbose=False)
    test_env.close()
    assert np.sum(test_results) == 8.0
    print("PybulletBlocksEnv integration test passed.")

    print("Integration tests passed.")


if __name__ == "__main__":
    import time
    start_time = time.time()
    test_ndr()
    test_ndr_set()
    test_planning()
    test_integration1()
    test_integration2()
    test_integration3()
    test_integration4()
    test_integration5()
    test_integration6()
    test_integration7()
    test_system()
    print("Tests completed in {} seconds".format(time.time() - start_time))
