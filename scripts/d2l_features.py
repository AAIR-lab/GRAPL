
import argparse
import pathlib
import sys
import tqdm


root = pathlib.Path(pathlib.Path(__file__).parent) / "../"
fd_root_path = root / "dependencies" / "fast-downward_stochastic"
d2l_root_path = root / "dependencies" / "d2l"
sltp_root_path = d2l_root_path / "src"
tarski_root_path = root / "dependencies" / "tarski"

def update_pythonpath():


    sys.path.append(root.as_posix())
    sys.path.append((root / "generalized_learning").as_posix())

    sys.path.append(fd_root_path.as_posix())
    sys.path.append(sltp_root_path.as_posix())

    tarski_src_path = tarski_root_path / "src"
    sys.path.append(tarski_src_path.as_posix())


update_pythonpath()

D2L_FEATUREGEN_PATH = (sltp_root_path / "generators").as_posix()

import d2l_sampler
import rddl_to_pddl_converter

from sltp import sampling
from sltp import featuregen
import types

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="D2L feature generator")
    
    parser.add_argument("--domain-file",
                        required=True,
                        help="The path to the domain file")
    
    parser.add_argument("--problem-file",
                        required=True,
                        help="The path to the problem file")
    
    parser.add_argument("--num-episodes",
                        type=int,
                        default=250,
                        help="The total episodes to sample for")
    
    parser.add_argument("--num-transitions",
                        type=int,
                        default=500,
                        help="The total transitions to sample each episode")
    
    parser.add_argument("--max-concept-size", type=int, default=5,
                        help="The max concept size to use")
    
    args = parser.parse_args()

    simulator_type = "pddl"
    
    domain_filepath = pathlib.Path(args.domain_file)
    problem_filepath = pathlib.Path(args.problem_file)
    
    assert domain_filepath.parent == problem_filepath.parent
    if domain_filepath.name.endswith(".rddl"):
        
        domain_filepath = rddl_to_pddl_converter.clean_rddl_file(
            domain_filepath.parent, domain_filepath.name, False)
        
        problem_filepath = rddl_to_pddl_converter.clean_rddl_file(
            problem_filepath.parent, problem_filepath.name, False)
        
        simulator_type = "rddl"
    
    samples_filepath = d2l_sampler.sample_d2l_states(
        domain_filepath, problem_filepath, args.num_episodes, 
        args.num_transitions, simulator_type)
    
    config = types.SimpleNamespace()
    config.sample_files = [samples_filepath]
    config.create_goal_features_automatically = False
    config.num_sampled_states = None
    
    # If RDDL, then no goals.
    if simulator_type == "rddl":
        config.parameter_generator = lambda _: []
        has_goals = False
    else:
        config.parameter_generator = None
        has_goals = True
        
    sample = sampling.sample_generated_states(config, None, has_goals=has_goals)
    sampling.log_sampled_states(sample, "%s.io" % (samples_filepath))
    
    transitions_info_filepath = "%s/transitions-info.io" % (
        domain_filepath.parent)
    sampling.print_transition_matrix(sample, transitions_info_filepath)
    
    config.domain = domain_filepath
    config.instances = [problem_filepath]
    config.concept_generator = None
    config.feature_generator = None
    
    config.experiment_dir = domain_filepath.parent
    config.generators_path = D2L_FEATUREGEN_PATH
    
    config.concept_generation_timeout = 120
    config.max_concept_size = args.max_concept_size
    config.distance_feature_max_complexity = args.max_concept_size
    config.cond_feature_max_complexity = 0
    config.comparison_features = False
    config.generate_goal_concepts = True
    config.print_denotations = False
    
    config.feature_matrix_filename = "%s/feature-matrix.data" % (
        domain_filepath.parent)
    features = featuregen.generate_feature_pool(config, sample)
    
    # Finally, clean all Star features.
    features_filepath = "%s/serialized-features.io" % (domain_filepath.parent)
    final_features_filepath = "%s/features.io" % (domain_filepath.parent)
    with open(features_filepath, "r") as features_fh:
        with open(final_features_filepath, "w") as final_features_fh:
            for line in features_fh:
                
                line = line.strip()
                if "Star" not in line:
                    final_features_fh.write(line + "\n")
    