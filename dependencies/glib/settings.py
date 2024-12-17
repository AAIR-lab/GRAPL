"""Settings used throughout the directory.
"""


class EnvConfig:
    """Environment-specific constants.
    """
    # domain_name = "Glibblocks"
    # domain_name = "Easygripper"
    # domain_name = "Glibdoors"
    # domain_name = "Tireworld"
    # domain_name = "Explodingblocks"
    domain_name = "PybulletBlocks"
    seed = 0

    # Number of test problems. Only needed for non-PDDLGym envs.
    num_test_problems = {}

    # Number of transitions to use for variational distance computation.
    num_var_dist_trans = {
        "Blocks": 1000,
        "Glibblocks": 1000,
        "Tsp": 1000,
        "Rearrangement": 1000,
        "Glibrearrangement": 1000,
        "Easygripper": 1000,
        "Doors": 1000,
        "Glibdoors": 1000,
        "Tireworld": 1000,
        "Tireworld_og": 1000,
        "Probabilistic_elevators": 1000,
        "First_responders": 1000,
        "Explodingblocks": 1000,
        "Explodingblocks_og": 1000,
        "PybulletBlocks": 10,
        "River": 1000,
        "NDRBlocks": 100,
        "Lavaworld" : 2000,
        "Roomgrid" : 1000,
        "Taxiworld" : 1000,


    }


class AgentConfig:
    """Agent-specific constants.
    """
    curiosity_methods_to_run = [
        "GLIB_L2",
        "GLIB_G1",
        "oracle",
        "random",
    ]
    cached_results_to_load = [
        # "GLIB_L2",
        # "GLIB_G1",
        # "oracle",
        # "random",
    ]
    # learning_name = "TILDE"
    learning_name = "LNDR"
    # learning_name = "groundtruth-PDDLEnv"+EnvConfig.domain_name+"-v0"
    planner_name = {
        "Blocks": "ff",
        "Glibblocks": "ff",
        "Tsp": "ff",
        "Rearrangement": "ff",
        "Easygripper": "ff",
        "Glibrearrangement": "ff",
        "Doors": "ff",
        "Glibdoors": "ff",
        "NDRBlocks": "ffreplan",
        "Tireworld": "ffreplan",
        "Tireworld_og": "ffreplan",
        "PybulletBlocks": "ffreplan",
        "Probabilistic_elevators": "ffreplan",
        "First_responders": "ffreplan",
        "Explodingblocks": "ffreplan",
        "Explodingblocks_og": "ffreplan",
        "River": "ffreplan",
        "Lavaworld" : "ffreplan",
        "Roomgrid" : "ffreplan",
        "Taxiworld" : "ffreplan",
    }

    # Random seed optionally used by curiosity modules.
    seed = 0
    # How often to learn operators.
    learning_interval = {
        "Blocks": 1,
        "Glibblocks": 1,
        "Tsp": 1,
        "Rearrangement": 1,
        "Glibrearrangement": 1,
        "Easygripper": 1,
        "Doors": 1,
        "Glibdoors": 1,
        "Tireworld": 10,
        "Tireworld_og": 10,
        "Probabilistic_elevators": 10,
        "First_responders": 10,
        "Explodingblocks": 10,
        "Explodingblocks_og": 10,
        "PybulletBlocks": 10,
        "River": 10,
        "NDRBlocks": 25,
        "Lavaworld" : 25,
        "Roomgrid" : 25,
        "Taxiworld" : 25,
    }

    # Max training episode length.
    max_train_episode_length = {
        "Blocks": 25,
        "Glibblocks": 25,
        "Tsp": 25,
        "Rearrangement": 25,
        "Glibrearrangement": 25,
        "Easygripper": 25,
        "Doors": 25,
        "Glibdoors": 25,
        "Tireworld": 40,
        "Tireworld_og": 8,
        "Probabilistic_elevators": 40,
        "First_responders": 40,
        "Explodingblocks": 40,
        "Explodingblocks_og": 25,
        "PybulletBlocks": 25,
        "River": 25,
        "PybulletBlocks" : 10,
        "NDRBlocks" : 25,
        "Lavaworld" : 300,
        "Roomgrid" : 300,
        "Taxiworld" : 300,
    }
    # Max test episode length.
    max_test_episode_length = {
        "Blocks": 25,
        "Glibblocks": 25,
        "Tsp": 25,
        "Rearrangement": 25,
        "Glibrearrangement": 25,
        "Easygripper": 100,
        "Doors": 25,
        "Glibdoors": 25,
        "Tireworld": 25,
        "Tireworld_og": 25,
        "Probabilistic_elevators": 100,
        "First_responders": 100,
        "Explodingblocks": 25,
        "Explodingblocks_og": 25,
        "River": 25,
        "PybulletBlocks" : 25,
        "NDRBlocks" : 25,
        "Lavaworld" : 300,
        "Roomgrid" : 300,
        "Taxiworld" : 300,
    }
    # Timeout for planner.
    planner_timeout = None  # set in main.py

    # Number of training iterations.
    num_train_iters = {
        "Blocks": 501,
        "Glibblocks": 501,
        "Tsp": 501,
        "Rearrangement": 1501,
        "Glibrearrangement": 1501,
        "Easygripper": 3001,
        "Doors": 2501,
        "Glibdoors": 2501,
        "Tireworld": 10000,
        "Tireworld_og": 401,
        "Probabilistic_elevators": 10000,
        "First_responders": 10000,
        "Explodingblocks": 10000,
        "Explodingblocks_og": 501,
        "River": 1001,
        "PybulletBlocks" : 501,
        "NDRBlocks" : 1501,
        "Lavaworld" : 1501,
        "Roomgrid" : 1501,
        "Taxiworld" : 1501,
    }

    ## Constants for curiosity modules. ##
    max_sampling_tries = 100
    max_planning_tries = 50
    oracle_max_depth = 2

    ## Constants for mutex detection. ##
    mutex_num_episodes = {
        "Blocks": 35,
        "Glibblocks": 35,
        "Tsp": 10,
        "Rearrangement": 35,
        "Glibrearrangement": 35,
        "Easygripper": 35,
        "Doors": 35,
        "Glibdoors": 35,
        "Tireworld": 35,
        "Tireworld_og": 35,
        "Probabilistic_elevators": 35,
        "First_responders": 35,
        "Explodingblocks": 35,
        "Explodingblocks_og": 35,
        "River": 35,
        "PybulletBlocks": 35,
        "NDRBlocks": 35,
        "Lavaworld" : 50,
        "Roomgrid" : 50,
        "Taxiworld" : 50,
    }
    mutex_episode_len = {
        "Blocks": 35,
        "Glibblocks": 35,
        "Tsp": 10,
        "Rearrangement": 35,
        "Glibrearrangement": 35,
        "Easygripper": 35,
        "Doors": 35,
        "Glibdoors": 35,
        "Tireworld": 35,
        "Tireworld_og": 35,
        "Probabilistic_elevators": 35,
        "First_responders": 35,
        "Explodingblocks": 35,
        "Explodingblocks_og": 35,
        "River": 35,
        "PybulletBlocks": 35,
        "NDRBlocks": 35,
        "Lavaworld" : 100,
        "Roomgrid" : 50,
        "Taxiworld" : 50,
    }
    mutex_num_action_samples = 10

    ## Constants for TILDE (also called FOLDT throughout code). ##
    max_foldt_feature_length = 10e8
    max_foldt_learning_time = 180
    max_foldt_exceeded_strategy = "fail" # 'fail' or 'early_stopping' or 'pdb'

    ## Constants for LNDR (also called ZPK throughout code). ##
    max_zpk_learning_time = 180
    max_zpk_explain_examples_transitions = {
        "Blocks": 25,
        "Glibblocks": 25,
        "Tsp": 25,
        "Rearrangement": 25,
        "Glibrearrangement": 25,
        "Easygripper": 25,
        "Doors": 25,
        "Glibdoors": 25,
        "Tireworld": float("inf"),
        "Tireworld_og": float("inf"),
        "Probabilistic_elevators": 25,
        "First_responders": 25,
        "Explodingblocks": 25,
        "Explodingblocks_og": 25,
        "River": 25,
        "PybulletBlocks": float("inf"),
        "NDRBlocks": float("inf"),
        "Lavaworld" : float("inf"),
        "Roomgrid" : float("inf"),
        "Taxiworld" : float("inf"),
    }
    max_zpk_action_batch_size = {
        "Blocks": None,
        "Glibblocks": None,
        "Tsp": None,
        "Rearrangement": None,
        "Glibrearrangement": None,
        "Easygripper": None,
        "Doors": None,
        "Glibdoors": None,
        "Tireworld": None,
        "Tireworld_og": None,
        "Probabilistic_elevators": None,
        "First_responders": None,
        "Explodingblocks": None,
        "Explodingblocks_og": None,
        "River": None,
        "PybulletBlocks": None,
        "NDRBlocks": None,
        "Lavaworld" : None,
        "Roomgrid" : None,
        "Taxiworld" : None,
    }
    zpk_initialize_from_previous_rule_set = {
        "Blocks": False,
        "Glibblocks": False,
        "Tsp": False,
        "Rearrangement": False,
        "Glibrearrangement": False,
        "Easygripper": False,
        "Doors": False,
        "Glibdoors": False,
        "Tireworld": True,
        "Tireworld_og": True,
        "Probabilistic_elevators": False,
        "First_responders": None,
        "Explodingblocks": False,
        "Explodingblocks_og": False,
        "River": False,
        "PybulletBlocks": False,
        "NDRBlocks": False,
        "Lavaworld" : False,
        "Roomgrid" : False,
        "Taxiworld" : False,

    }

    # Major hacks. Only used by oracle_curiosity.py.
    train_env = None


class GeneralConfig:
    """General configuration constants.
    """
    verbosity = 5
    num_seeds = 10
