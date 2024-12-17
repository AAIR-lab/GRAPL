"""Utilities
"""
from PIL import Image

from collections import defaultdict

import itertools
import numpy as np
import os
import gym
import imageio
import sys
import contextlib


class DummyFile:
    """Helper for nostdout().
    """
    def write(self, x):
        """Dummy write method.
        """
        pass

    def flush(self):
        """Dummy flush method.
        """
        pass


@contextlib.contextmanager
def nostdout():
    """Context for suppressing output. Usage:
    import nostdout
    with nostdout():
        foo()
    """
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


def get_env_id(env):
    try:
        return env.spec.id
    except AttributeError:
        return env.__class__.__name__

def run_policy(env, policy, max_num_steps=10, verbose=False, check_reward=True, render=True, 
               outdir='/tmp/', fps=3):
    if outdir is None:
        outdir = "/tmp/{}".format(get_env_id(env))
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    if render:
        video_path = os.path.join(outdir, 'policy_{}_demo.gif'.format(get_env_id(env)))
        env = VideoWrapper(env, video_path, fps=fps)

    obs, debug_info = env.reset()

    tot_reward = 0.
    for t in range(max_num_steps):
        if verbose:
            print("Obs:", obs)
    
        action = policy(obs)
        if verbose:
            print("Act:", action)

        obs, reward, done, debug_info = env.step(action)
        if render:
            env.render()
        tot_reward += reward
        if verbose:
            print("Rew:", reward)

        if done:
            break

    if check_reward:
        assert tot_reward > 0
    return tot_reward


class VideoWrapper(gym.Wrapper):
    def __init__(self, env, out_path, fps=30, size=None):
        super().__init__(env)
        self.out_path_prefix = '.'.join(out_path.split('.')[:-1])
        self.out_path_suffix = out_path.split('.')[-1]
        self.fps = fps
        self.size = size
        self.reset_count = 0
        self.images = []

    def reset(self):
        if len(self.images) > 0:
            self._finish_video()

        obs = super().reset()

        # Handle problem-dependent action spaces
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.out_path = self.out_path_prefix + str(self.reset_count) + \
            '.' + self.out_path_suffix
        self.reset_count += 1

        self.images = []
        img = super().render()
        img = self.process_image(img)
        self.images.append(img)

        return obs

    def step(self, action):
        obs, reward, done, debug_info = super().step(action)

        img = super().render()
        img = self.process_image(img)
        self.images.append(img)

        return obs, reward, done, debug_info

    def close(self):
        if len(self.images) > 0:
            self._finish_video()
        return super().close()

    def process_image(self, img):
        if self.size is None:
            return img
        return np.array(Image.fromarray(img).resize(self.size), dtype=img.dtype)

    def _finish_video(self):
        imageio.mimsave(self.out_path, self.images, fps=self.fps)
        print("Wrote out video to {}.".format(self.out_path))

