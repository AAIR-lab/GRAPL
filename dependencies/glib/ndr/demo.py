from planning import find_policy
from utils import run_random_agent_demo, run_plan, run_policy
from envs.ndr_blocks import NDRBlocksEnv, pickup, puton, putontable
from envs.pybullet_blocks import PybulletBlocksEnv
import os

def run_all(render=True, verbose=True):
    env = NDRBlocksEnv()
    env.seed(0)
    # env = PybulletBlocksEnv()
    # initial_state, debug_info = env.reset()
    # goal = debug_info["goal"]
    # policy = find_policy("ff_replan", initial_state, goal, env.operators, env.action_space, env.observation_space)
    # total_returns = 0
    # for trial in range(10):
    #     outdir = '/tmp/ndrblocks{}/'.format(trial)
    #     os.makedirs(outdir, exist_ok=True)
    #     returns = run_policy(env, policy, verbose=verbose, render=render, check_reward=False, 
    #         outdir=outdir)
    #     total_returns += returns
    # print("Average returns:", total_returns/10.)
    plan = [pickup("a"), puton("b"), pickup("c"), puton("a")]
    run_plan(env, plan, verbose=verbose, render=render)
    # env = PybulletBlocksEnv(record_low_level_video=True, video_out='/tmp/lowlevel_plan_example.gif')
    # env.seed(5)
    # run_random_agent_demo(env, verbose=verbose, seed=1, render=render)
    # plan = [pickup("block1"), puton("block2"), pickup("block0"), puton("block1")]
    # plan = [pickup("block1"), putontable(), pickup("block2"), putontable(), pickup("block0"), putontable()]
    # run_plan(env, plan, verbose=verbose, render=render)


if __name__ == '__main__':
    run_all(render=True)
