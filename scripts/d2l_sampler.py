
import argparse
import pathlib
import sys
import tqdm


def update_pythonpath():

    root = pathlib.Path(pathlib.Path(__file__).parent) / "../"

    sys.path.append(root.as_posix())
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

from generalized_learning.concretized import problem as problem_instance
from generalized_learning.util import state_explorer


def _get_predicate_str(predicate):

    string = "%s(" % (predicate.predicate)

    for arg in predicate.args:

        string += ",%s" % (arg)

    string = string.replace("(,", "(")

    string += ")"
    return string


def write_d2l_sampled_states(output_filepath, problem, state_id_map,
                             transition_set,
                             unsupported_predicates=set(["="])):

    file_handle = open(output_filepath, "w")

    for state in state_id_map:

        idx = state_id_map[state]

        atom_str = ""
        for predicate in state.get_atom_set():

            if predicate.predicate not in unsupported_predicates:

                atom_str += " %s" % (_get_predicate_str(predicate))

        atom_str = atom_str.strip()
        
        if not problem.has_goals():
            file_handle.write("(N) %u 1 0 %s\n" % (idx, atom_str))
        elif problem.is_goal_satisfied(state):
            file_handle.write("(N) %u 1 0 %s\n" % (idx, atom_str))
        else:
            file_handle.write("(N) %u 0 0 %s\n" % (idx, atom_str))
            
    for id_1, id_2 in transition_set:

        file_handle.write("(E) %u %u\n" % (id_1, id_2))

    file_handle.close()

def sample_d2l_states(domain_filepath, problem_filepath, num_episodes, 
                      num_transitions, simulator_type="generic"):

    assert domain_filepath.parent == problem_filepath.parent

    problem = problem_instance.create_problem(domain_filepath.name,
                                              problem_filepath.name,
                                              domain_filepath.parent,
                                              simulator_type)

    output_filepath = "%s/%s.sampled" % (problem_filepath.parent,
                                         problem_filepath.name)

    state_id_map = {}
    transition_set = set()

    progress_bar = tqdm.tqdm(total=num_episodes,
                             unit=" episodes")

    for _ in range(num_episodes):

        progress_bar.update(1)

        transitions = state_explorer.sample_transitions(
            problem,
            num_transitions,
            disable_progress_bar=True)

        for s1, s2 in transitions:

            id_1 = state_id_map.setdefault(s1, len(state_id_map))
            id_2 = state_id_map.setdefault(s2, len(state_id_map))

            transition_set.add((id_1, id_2))

    progress_bar.close()

    print("Total states...........: %u" % (len(state_id_map)))
    print("Total transitions......: %u" % (len(transition_set)))

    print("Writing samples to", output_filepath)
    write_d2l_sampled_states(output_filepath, problem, state_id_map, 
                             transition_set)
    
    return output_filepath

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-file", required=True,
                        help="The input domain file.")
    parser.add_argument("--problem-file", required=True,
                        help="The input problem file")
    parser.add_argument("--simulator-type", type=str, default="generic",
                        choices=["generic", "rddl"],
                        help="The simulator type.")
    parser.add_argument("--num-episodes", default=250, type=int,
                        help="The total number of episodes to run")
    parser.add_argument("--num-transitions", default=500, type=int,
                        help="Total transitions to sample per episode")

    args = parser.parse_args()

    domain_filepath = pathlib.Path(args.domain_file)
    problem_filepath = pathlib.Path(args.problem_file)

    sample_d2l_states(domain_filepath, problem_filepath, args.num_episodes, 
                      args.num_transitions, args.simulator_type)

