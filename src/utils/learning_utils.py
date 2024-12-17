#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import pprint
import copy
from config import *

from pddlgym.structs import ProbabilisticEffect
def generate_samples(domain_name, env, agent):
    """
    Based on GLIB's transition generation
    """
    file_name = "../temp_files/{}.".format(domain_name)

    if os.path.exists(file_name):
        with open(file_name, "rb") as f:
            transitions = pickle.load(f)
       # return transitions

    #Actions with actual preconditions and effects
    fullActionSpecs = agent.actions

    actions = env.action_space.predicates
    total_counts = {a: 0 for a in actions}
    actionPrecon = dict()
    num_no_effects = {a: 0 for a in actions}
    transitions = []
    num_trans_per_act = 100
    num_problems = len(env.problems)
    while True:
        if all(c >= num_trans_per_act for c in total_counts.values()):
            break
        obs, _ = env.reset()
        for _1 in range(transition_count // num_problems):
            action = env.action_space.sample(obs)
            next_obs, _, done, _ = env.step(action)
            null_effect = (next_obs.literals == obs.literals)
            keep_transition = ((not null_effect or
                                (num_no_effects[action.predicate] <
                                 total_counts[action.predicate] / 2 + 1)) and
                               total_counts[action.predicate] < num_trans_per_act)
            if keep_transition:
                total_counts[action.predicate] += 1
                if null_effect:
                    num_no_effects[action.predicate] += 1
                transitions.append((obs, action, next_obs))
            if done:
                break
            obs = next_obs

    #Get all actions attempted to be executed.
    concreteActions = []
    #print(transitions[1][1].predicate)
    for t in transitions:
        concreteActions.append(t[1])
    concreteActions = set(concreteActions)

    #Remove all literals not seen prior to action execution.
    unknownPre = set()
    for a in concreteActions:
        foundInstance = False
        result = frozenset()
        numOccurences = 0
        for t in transitions:
            actualEffect = t[0].literals != t[2].literals
            if a == t[1] and t[0].literals != t[2].literals:
                if not foundInstance:
                    result = t[0].literals
                    foundInstance = True
                else:
                    result = result & t[0].literals
                    if result == frozenset():
                        print('This is not possible.')
        actionPrecon[a] = result

    #Remove all literals whose arguments are not a subset of the action.
    for a in actionPrecon:
        removePre = []
        for l in actionPrecon[a]:
            notSubset = False
            for v in l.variables:
                if v not in a.variables:
                    notSubset = True
                    break
            if notSubset:
                removePre.append(l)
        removePre = frozenset(removePre)
        actionPrecon[a] = actionPrecon[a] - removePre
    for a in actionPrecon:
        if actionPrecon[a] == frozenset():
            unknownPre.add(a)
    for a in unknownPre:
        del actionPrecon[a]


    actionNameToPars, generalPredicates, actionNameAndPars = actionVarDicts(fullActionSpecs)

    #Maps parameterized actions to concrete counterparts.
    generalToSpec = dict()
    for g in actionNameToPars:
        generalToSpec[g] = []
    for a in actionPrecon:
        for g in actionNameToPars:
            if g == a.predicate:
                generalToSpec[g].append(a)
                break

    for g in actionNameToPars:
        for p in generalPredicates:
            if g in p:
                generalToSpec[p] = generalToSpec[g]
                del generalToSpec[g]
                break

    #Show generalized preconditions
    actionPrecon = generalizePreconditions(generalToSpec, actionPrecon, generalPredicates)
    #Show whether or not the preconditions predicted are a subset of the actual preconditions.
    comparePrecon(fullActionSpecs, actionNameAndPars, actionPrecon)


    pp = pprint.PrettyPrinter(indent=4)
    with open(file_name, "wb") as f:
        pp.pprint(actionPrecon)

        pickle.dump(transitions, f)
    return transitions

def getConcreteVars(actionPrecon):
    actsAndVars = {}
    for a in actionPrecon:
        variablesForA = str(a).split('()')
        if len(variablesForA) == 2:
            continue
        else:
            variablesForA = ''.join(variablesForA)
            variablesForA = variablesForA.split('(')
            variablesForA = variablesForA[1:]
            variablesForA = ''.join(variablesForA)
            variablesForA = variablesForA.split(')')
            variablesForA = variablesForA[:-1]
            variablesForA = ''.join(variablesForA)
            variablesForA = variablesForA.split(',')
        actsAndVars[a] = variablesForA
    return actsAndVars

def actionVarDicts(fullActionSpecs):
    generalPredicates = dict()
    actionNameAndPars = dict()
    actionNameToPars = dict()
    for a in fullActionSpecs:
        general = str(fullActionSpecs[a]).split('):')[0] + ')'
        generalPredicates[general] = []
        generalNameOnly = general.split('(')[0]
        actionNameAndPars[general] = generalNameOnly

    # Map
    for a in generalPredicates:
        variablesForA = a.split('()')
        if len(variablesForA) == 2:
            continue
        else:
            variablesForA = ''.join(variablesForA)
            variablesForA = variablesForA.split('(')
            variablesForA = variablesForA[1:]
            variablesForA = ''.join(variablesForA)
            variablesForA = variablesForA.split(')')
            variablesForA = variablesForA[:-1]
            variablesForA = ''.join(variablesForA)
            variablesForA = variablesForA.split(',')
        generalPredicates[a] = variablesForA
        actionNameToPars[actionNameAndPars[a]] = variablesForA
    return actionNameToPars, generalPredicates, actionNameAndPars

def replaceConcreteWithGeneral(actionPreconGen, actsAndVars, generalToSpec, generalPredicates):
    for g in generalPredicates:
        for a in generalToSpec[g]:
            numPrecs = len(actionPreconGen[a])
            i = 0
            while(i < numPrecs):
                numVars = len(actsAndVars[a])
                j = 0
                while(j < numVars):
                    #print(type(generalPredicates[g][j]))
                    actionPreconGen[a][i] = actionPreconGen[a][i].replace(actsAndVars[a][j], generalPredicates[g][j])
                    j += 1
                i += 1
            actionPreconGen[a] = frozenset(actionPreconGen[a])
    return actionPreconGen

def generalizePreconditions(generalToSpec, actionPrecon, generalPredicates):
    actionPreconGen = copy.copy(actionPrecon)
    for a in actionPreconGen:
        actionPreconGen[a] = list(actionPreconGen[a])
        numPrecon = len(actionPreconGen[a])
        i = 0
        while(i < numPrecon):
            actionPreconGen[a][i] = str(actionPreconGen[a][i])
            i += 1

    actsAndVars = getConcreteVars(actionPreconGen)

    transformedPreconditions = replaceConcreteWithGeneral(actionPreconGen,actsAndVars,generalToSpec,generalPredicates)

    parameterizedPre = {g:frozenset() for g in generalPredicates}

    for g in generalPredicates:
        gFound = False
        for a in generalToSpec[g]:
            if not gFound:
                parameterizedPre[g] = transformedPreconditions[a]
                gFound = True
            else:
                parameterizedPre[g] &= transformedPreconditions[a]


    return parameterizedPre

def comparePrecon(trueActionSpecs, actionNameAndPars, expectedPrecons):
    truePrecons = dict()
    namesToPar = {a: p for p, a in actionNameAndPars.items()}
    for a in trueActionSpecs:

        aPars = namesToPar[a]
        truePrecons[aPars] = str(trueActionSpecs[a])
        truePrecons[aPars] = truePrecons[aPars].split(' ')
        truePrecons[aPars] = truePrecons[aPars][1:]
        truePrecons[aPars] = ''.join(truePrecons[aPars])
        truePrecons[aPars] = truePrecons[aPars].split('=>')
        truePrecons[aPars] = truePrecons[aPars][0]
        truePrecons[aPars] = truePrecons[aPars].split('&')
        truePrecons[aPars] = frozenset(truePrecons[aPars])

    isSubset = True

    for a in truePrecons:
        if not (expectedPrecons[a] <= truePrecons[a]):
            isSubset = False
            break
    print(isSubset)

def extract_elements(env, problem_idx=None):

    assert problem_idx is not None or len(env.problems) == 1
    problem_idx = 0 if problem_idx is None else problem_idx
    return env.domain, env.problems[problem_idx]


def abstract_action(actions):
    new_actions = copy.deepcopy(actions)
    for _a in new_actions:
        new_actions[_a].effects.literals = []
        
        if isinstance(new_actions[_a].effects, ProbabilisticEffect):
            
            new_actions[_a].effects.probabilities = []


        # Comment for testing of effect learning
        new_actions[_a].preconds.literals = []

    return new_actions