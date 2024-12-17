#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import glob
import pickle
import subprocess

from config import *
from lattice import State
from utils import generate_ds, set_to_state

# DOMAIN = "barman"


def get_state_from_val(domain_file, problem_file, plan_file, objects, init_state):
    param = VAL_PATH + " -v " + domain_file + " " + problem_file + " " + plan_file + " > " + GEN_VAL_FILE
    p = subprocess.Popen([param], shell=True)
    p.wait()
    all_states = []

    curr_state = set(copy.deepcopy(init_state))

    # Scanning the VAL output
    f = open(GEN_VAL_FILE, 'r')
    for line in f:
        # Adding a new policy rule
        if "Checking next happening (time " in line or "Plan executed successfully" in line:
            _state = State(set_to_state(curr_state), objects)
            all_states.append(copy.deepcopy(_state))

        if "Deleting " in line:
            pred = line.replace("Deleting ", "").replace("(", "").replace(")\n", "").split(" ")
            pred = "|".join(pred)
            curr_state.discard(pred)

        if "Adding " in line:
            pred = line.replace("Adding ", "").replace("(", "").replace(")\n", "").split(" ")
            pred = "|".join(pred)
            curr_state.add(pred)

    f.close()

    return all_states


def run_ff(domain_file, problem_file, output_file):
    param = FF_PATH + "ff"
    param += " -o " + domain_file
    param += " -f " + problem_file
    param += " -i 120"
    param += " > " + output_file

    p = subprocess.Popen([param], shell=True)
    p.wait()
    return


def main():
    for DOMAIN in domains:
        domain_file = DOMAINS_PATH + DOMAIN + "/" + DOMAIN_FILE
        problem_file_path = DOMAINS_PATH + DOMAIN + "/" + PROBLEM_DIR + "/*.pddl"
        problem_file_l = glob.glob(problem_file_path)
        problem_file_list = sorted(problem_file_l)
        all_states = []
        for problem_file in problem_file_list:
            _, _, _, _, objects, _, init_state, domain_name = generate_ds(domain_file, problem_file)

            run_ff(domain_file, problem_file, GEN_RESULT_FILE)
            f = open(GEN_RESULT_FILE, "r")
            _plan_found = False
            _plan = ""
            step = 0
            for x in f:
                if "found legal plan as follows" in x:
                    _plan_found = True
                if not _plan_found:
                    continue

                if str(step) + ":" in x:
                    k = copy.deepcopy(x)
                    _plan += str(step) + " : (" + k.lower().rstrip().split(":")[-1].lstrip() + ")\n"
                    step += 1

                if "time spent" in x:
                    break
            f.close()

            f = open(GEN_PLAN_FILE, "w")
            f.write(_plan)
            f.close()

            states = get_state_from_val(domain_file, problem_file, GEN_PLAN_FILE, objects, init_state)
            all_states.extend(states)
            if len(all_states) >= STORED_STATES_COUNT:
                break

        with open(RANDOM_STATE_FOLDER + "random_" + DOMAIN + ".pkl", "wb") as f:
            pickle.dump(all_states[0:STORED_STATES_COUNT+1], f)


if __name__ == "__main__":
    main()
