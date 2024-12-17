"""A blocks environment written in Pybullet.

Based on the environment described in ZPK.
"""
from pddlgym.structs import Predicate, LiteralConjunction, Type, Anti, State
from pddlgym.spaces import LiteralSpace, LiteralSetSpace
from glib.ndr.utils import VideoWrapper
import random

from gym import utils, spaces
from gym.utils import seeding
import gym
import numpy as np
import itertools

import pybullet as p

import glob
import os


DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DEBUG = False

### First set up an environment that is just the low-level physics
def inverse_kinematics(body_id, end_effector_id, target_position, target_orientation, joint_indices, physics_client_id=-1):
    """
    Parameters
    ----------
    body_id : int
    end_effector_id : int
    target_position : (float, float, float)
    target_orientation : (float, float, float, float)
    joint_indices : [ int ]
    
    Returns
    -------
    joint_poses : [ float ] * len(joint_indices)
    """
    lls, uls, jrs, rps = get_joint_ranges(body_id, joint_indices, physics_client_id=physics_client_id)

    all_joint_poses = p.calculateInverseKinematics(body_id, end_effector_id, target_position,
        targetOrientation=target_orientation,
        lowerLimits=lls, upperLimits=uls, jointRanges=jrs, restPoses=rps,
        physicsClientId=physics_client_id)

    # Find the free joints
    free_joint_indices = []

    num_joints = p.getNumJoints(body_id, physicsClientId=physics_client_id)
    for idx in range(num_joints):
        joint_info = p.getJointInfo(body_id, idx, physicsClientId=physics_client_id)
        if joint_info[3] > -1:
            free_joint_indices.append(idx)

    # Find the poses for the joints that we want to move
    joint_poses = []

    for idx in joint_indices:
        free_joint_idx = free_joint_indices.index(idx)
        joint_pose = all_joint_poses[free_joint_idx]
        joint_poses.append(joint_pose)

    return joint_poses

def get_joint_ranges(body_id, joint_indices, physics_client_id=-1):
    """
    Parameters
    ----------
    body_id : int
    joint_indices : [ int ]

    Returns
    -------
    lower_limits : [ float ] * len(joint_indices)
    upper_limits : [ float ] * len(joint_indices)
    joint_ranges : [ float ] * len(joint_indices)
    rest_poses : [ float ] * len(joint_indices)
    """
    lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []

    num_joints = p.getNumJoints(body_id, physicsClientId=physics_client_id)

    for i in range(num_joints):
        joint_info = p.getJointInfo(body_id, i, physicsClientId=physics_client_id)

        # Fixed joint so ignore
        qIndex = joint_info[3]
        if qIndex <= -1:
            continue

        ll, ul = -2., 2.
        jr = 2.

        # For simplicity, assume resting state == initial state
        rp = p.getJointState(body_id, i, physicsClientId=physics_client_id)[0]

        # Fix joints that we don't want to move
        if i not in joint_indices:
            ll, ul =  rp-1e-8, rp+1e-8
            jr = 1e-8

        lower_limits.append(ll)
        upper_limits.append(ul)
        joint_ranges.append(jr)
        rest_poses.append(rp)

    return lower_limits, upper_limits, joint_ranges, rest_poses

def get_kinematic_chain(robot_id, end_effector_id, physics_client_id=-1):
    """
    Get all of the free joints from robot base to end effector.

    Includes the end effector.

    Parameters
    ----------
    robot_id : int
    end_effector_id : int
    physics_client_id : int

    Returns
    -------
    kinematic_chain : [ int ]
        Joint ids.
    """
    kinematic_chain = []
    while end_effector_id > 0:
        joint_info = p.getJointInfo(robot_id, end_effector_id, physicsClientId=physics_client_id)
        if joint_info[3] > -1:
            kinematic_chain.append(end_effector_id)
        end_effector_id = joint_info[-1]
    return kinematic_chain


class LowLevelPybulletBlocksEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 10}

    def __init__(self, use_gui=True, sim_steps_per_action=20, physics_client_id=None, max_joint_velocity=0.1):

        self.sim_steps_per_action = sim_steps_per_action
        self.max_joint_velocity = max_joint_velocity

        self.distance_threshold = 0.05

        self.base_position = [0.405 + 0.2869, 0.48 + 0.2641, 0.0]
        self.base_orientation = [0., 0., 0., 1.]

        self.table_height = 0.42 + 0.205

        self.camera_distance = 1.5
        self.yaw = 90
        self.pitch = -24
        self.camera_target = [1.65, 0.75, 0.42]

        if physics_client_id is None:
            if use_gui:
                self.physics_client_id = p.connect(p.GUI)
                p.resetDebugVisualizerCamera(self.camera_distance, self.yaw, self.pitch, self.camera_target)
            else:
                self.physics_client_id = p.connect(p.DIRECT)
        else:
            self.physics_client_id = physics_client_id

        self.use_gui = use_gui
        self.setup()
        self.seed()

    def sample_initial_state(self):
        """Sample blocks
        """
        # Block name : block state
        state = {}
        block_name_counter = itertools.count()

        # For now, constant orientation (quaternion) for all blocks.
        orn_x, orn_y, orn_z, orn_w = 0., 0., 0., 1.
        random.seed(931)
        total_num_blocks = 4
        num_blocks = 0
        num_piles = random.randint(1, 4)
        for pile in range(num_piles):
            if num_blocks == total_num_blocks:
                break
            if pile == num_piles-1:
                num_blocks_in_pile = total_num_blocks - num_blocks
            else:
                num_blocks_in_pile = random.randint(1, total_num_blocks-num_blocks+1)
            num_blocks += num_blocks_in_pile
            # Block stack blocks.
            x, y = 1.25, 0.5 + pile*0.2
            previous_block_top = 0.5
            for i in range(num_blocks_in_pile):
                block_name = "block{}".format(next(block_name_counter))
                w, l, h, mass, friction = self.sample_block_static_attributes()
                z = previous_block_top + h/2.
                previous_block_top += h
                attributes = [w, l, h, x, y, z, orn_x, orn_y, orn_z, orn_w, mass, friction]
                state[block_name] = attributes

        return state

    def sample_block_static_attributes(self):
        w, l, h = 0.075, 0.075, 0.075
        mass = 0.05
        friction = 1.2
        # w, l, h = self.np_random.normal(0.075, 0.005, size=(3,))
        # mass = self.np_random.uniform(0.05, 0.05)
        # friction = 1.
        return w, l, h, mass, friction

    def setup(self):
        p.resetSimulation(physicsClientId=self.physics_client_id)

        # Load plane
        p.setAdditionalSearchPath(DIR_PATH)
        p.loadURDF("assets/urdf/plane.urdf", [0, 0, -1], useFixedBase=True, physicsClientId=self.physics_client_id)

        # Load Fetch
        self.fetch_id = p.loadURDF("assets/urdf/robots/fetch.urdf", useFixedBase=True, physicsClientId=self.physics_client_id)
        p.resetBasePositionAndOrientation(self.fetch_id, self.base_position, self.base_orientation, physicsClientId=self.physics_client_id)

        # Get end effector
        joint_names = [p.getJointInfo(self.fetch_id, i, physicsClientId=self.physics_client_id)[1].decode("utf-8") \
                       for i in range(p.getNumJoints(self.fetch_id, physicsClientId=self.physics_client_id))]

        self.ee_id = joint_names.index('gripper_axis')
        self.ee_orientation = [1., 0., -1., 0.]

        self.arm_joints = get_kinematic_chain(self.fetch_id, self.ee_id, physics_client_id=self.physics_client_id)
        self.left_finger_id = joint_names.index("l_gripper_finger_joint")
        self.right_finger_id = joint_names.index("r_gripper_finger_joint")
        self.arm_joints.append(self.left_finger_id)
        self.arm_joints.append(self.right_finger_id)

        # Load table
        table_urdf = "assets/urdf/table.urdf"
        table_id = p.loadURDF(table_urdf, useFixedBase=True, physicsClientId=self.physics_client_id)
        p.resetBasePositionAndOrientation(table_id, (1.65, 0.75, 0.0), [0., 0., 0., 1.], physicsClientId=self.physics_client_id)

        # Blocks are created at reset
        self.block_ids = {}

        # Set gravity
        p.setGravity(0., 0., -10., physicsClientId=self.physics_client_id)

        # Let the world run for a bit
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.physics_client_id)

        # Move the arm to a good start location
        self._initial_joint_values = inverse_kinematics(self.fetch_id, self.ee_id, [1., 0.5, 0.5], self.ee_orientation, 
            self.arm_joints, physics_client_id=self.physics_client_id)

        # Set arm joint motors
        for joint_idx, joint_val in zip(self.arm_joints, self._initial_joint_values):
            p.resetJointState(self.fetch_id, joint_idx, joint_val, physicsClientId=self.physics_client_id)

        for _ in range(100):
            p.stepSimulation(physicsClientId=self.physics_client_id)

        # Record the initial state so we can reset to it later
        # this doesn't seem to work
        # self.initial_state_id = p.saveState(physicsClientId=self.physics_client_id)

        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def set_state(self, state):
        # Blocks are always recreated on reset because their size, mass, friction changes
        self.static_block_attributes = {}
        self.block_ids = {}

        for block_name, block_state in state.items():
            color = self.get_color_from_block_name(block_name)
            block_id = self.create_block(block_state, color=color)
            self.block_ids[block_name] = block_id

        # Let the world run for a bit
        for _ in range(250):
            p.stepSimulation(physicsClientId=self.physics_client_id)

    def get_color_from_block_name(self, block_name):
        colors = [
            (0.95, 0.05, 0.1, 1.),
            (0.05, 0.95, 0.1, 1.),
            (0.1, 0.05, 0.95, 1.),
            (0.4, 0.05, 0.6, 1.),
            (0.6, 0.4, 0.05, 1.),
            (0.05, 0.04, 0.6, 1.),
            (0.95, 0.95, 0.1, 1.),
            (0.95, 0.05, 0.95, 1.),
            (0.05, 0.95, 0.95, 1.),
        ]
        block_num = int(block_name[len("block"):])
        return colors[block_num % len(colors)]

    def create_block(self, attributes, color=(0., 0., 1., 1.)):
        w, l, h, x, y, z, orn_x, orn_y, orn_z, orn_w, mass, friction = attributes

        # Create the collision shape
        half_extents = [w/2., l/2., h/2.]
        collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.physics_client_id)

        # Create the visual_shape
        visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color, 
            physicsClientId=self.physics_client_id)

        # Create the body
        block_id = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_id, 
            baseVisualShapeIndex=visual_id, basePosition=[x, y, z], baseOrientation=[orn_x, orn_y, orn_z, orn_w],
            physicsClientId=self.physics_client_id)
        p.changeDynamics(block_id, -1, lateralFriction=friction, physicsClientId=self.physics_client_id)

        # Cache block static attributes
        self.static_block_attributes[block_id] = (w, l, h, mass, friction)

        return block_id

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        return [seed]

    def reset(self):
        for block_id in self.block_ids.values():
            p.removeBody(block_id, physicsClientId=self.physics_client_id)

        # Make sure the robot gets out of the way
        self.block_ids = {}
        for _ in range(10):
            self.step(np.array([-0.5, -0.5, 0.5, 0.0]))

        initial_state = self.sample_initial_state()
        self.set_state(initial_state)

        # With 50% probability, start out holding the last block, which must be on top
        if self.np_random.uniform() < 0.5:
            state = self.get_state()
            top_block = Type("block")(max(state["blocks"]))
            controller = controllers[pickup]
            controller.reset()
            for _ in range(100):
                a, c_done = controller.step([top_block], state)
                if c_done:
                    break
                state, _, done, _ = self.step(a)

        return self.get_state(), {}

    def step(self, action):
        action *= 0.05

        ee_delta, finger_action = action[:3], action[3]

        current_position, current_orientation = p.getLinkState(self.fetch_id, self.ee_id, physicsClientId=self.physics_client_id)[4:6]
        target_position = np.add(current_position, ee_delta)

        joint_values = inverse_kinematics(self.fetch_id, self.ee_id, target_position, self.ee_orientation, self.arm_joints, 
            physics_client_id=self.physics_client_id)

        # Set arm joint motors
        for joint_idx, joint_val in zip(self.arm_joints, joint_values):
            p.setJointMotorControl2(bodyIndex=self.fetch_id, jointIndex=joint_idx, controlMode=p.POSITION_CONTROL,
                targetPosition=joint_val, physicsClientId=self.physics_client_id)

        # Set finger joint motors
        for finger_id in [self.left_finger_id, self.right_finger_id]:
            current_val = p.getJointState(self.fetch_id, finger_id, physicsClientId=self.physics_client_id)[0]
            target_val = current_val + finger_action
            p.setJointMotorControl2(bodyIndex=self.fetch_id, jointIndex=finger_id, controlMode=p.POSITION_CONTROL,
                targetPosition=target_val, physicsClientId=self.physics_client_id)

        for _ in range(self.sim_steps_per_action):
            p.stepSimulation(physicsClientId=self.physics_client_id)

        obs = self.get_state()
        done = False
        info = {}
        reward = 0.

        return obs, reward, done, info

    def close(self):
        p.disconnect(self.physics_client_id)

    def get_state(self):
        gripper_position, gripper_velocity  = p.getLinkState(self.fetch_id, self.ee_id, physicsClientId=self.physics_client_id)[4:6]
        left_finger_pos = p.getJointState(self.fetch_id, self.left_finger_id, physicsClientId=self.physics_client_id)[0]

        obs = {
            'gripper' : [gripper_position, gripper_velocity, left_finger_pos],
            'blocks' : {},
        }

        for block_name, block_id in self.block_ids.items():
            obs['blocks'][block_name] = self.get_block_attributes(block_id)

        return obs

    def get_block_attributes(self, block_id):
        w, l, h, mass, friction = self.static_block_attributes[block_id]

        (x, y, z), (orn_x, orn_y, orn_z, orn_w) = p.getBasePositionAndOrientation(block_id, 
            physicsClientId=self.physics_client_id)

        attributes = [w, l, h, x, y, z, orn_x, orn_y, orn_z, orn_w, mass, friction]
        return attributes

    def render(self, mode='human', close=False):

        if not self.use_gui:
            raise Exception("Rendering only works with GUI on. See https://github.com/bulletphysics/bullet3/issues/1157")

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_distance,
            yaw=self.yaw,
            pitch=self.pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self.physics_client_id)

        width, height = 2*(3350//8), 2*(1800//8)

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(width / height),
            nearVal=0.1, farVal=100.0,
            physicsClientId=self.physics_client_id)

        (_, _, px, _, _) = p.getCameraImage(
            width=width, height=height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.physics_client_id)

        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array



# Object types
block_type = Type("block")

# Actions
pickup = Predicate("pickup", 1, [block_type])
puton = Predicate("puton", 1, [block_type])
putontable = Predicate("putontable", 0, [])

# Controllers
atol = 1e-4
def get_move_action(gripper_position, target_position, atol=1e-3, gain=5, max_vel_norm=1, close_gripper=False):
    """
    Move an end effector to a position and orientation.
    """
    # Get the currents
    action = gain * np.subtract(target_position, gripper_position)
    action_norm = np.linalg.norm(action)
    if action_norm > max_vel_norm:
        action = action * max_vel_norm / action_norm

    if close_gripper:
        gripper_action = -0.1
    else:
        gripper_action = 0.
    action = np.hstack((action, gripper_action))

    return action

def block_is_grasped(left_finger_pos, gripper_position, block_position, relative_grasp_position, atol=1e-3):
    block_inside = block_inside_grippers(gripper_position, block_position, relative_grasp_position, atol=atol)
    if DEBUG: print("block inside?", block_inside)
    grippers_closed = grippers_are_closed(left_finger_pos, atol=atol)
    if DEBUG: print("grippers_closed?", grippers_closed)
    return block_inside and grippers_closed

def block_inside_grippers(gripper_position, block_position, relative_grasp_position, atol=1e-3):
    relative_position = np.subtract(gripper_position, block_position)
    return np.sum(np.subtract(relative_position, relative_grasp_position)**2) < atol

def grippers_are_closed(left_finger_pos, atol=1e-3):
    return abs(left_finger_pos) - 0.02 <= atol

def grippers_are_open(left_finger_pos, atol=1e-3):
    return abs(left_finger_pos - 0.05) <= atol


class StateMachineController:
    stage = 0

    def reset(self):
        self.stage = 0

    def step(self, objects, obs):
        if self.stage == 0 and self.terminate_early(objects, obs):
            return None, True
        if self.stage > len(self.stages)-1:
            return None, True
        action, stage_done = self.stages[self.stage](objects, obs)
        if stage_done:
            self.stage += 1
        return action, self.stage > len(self.stages)-1

    def terminate_early(self, objects, obs):
        return False



class PickupController(StateMachineController):

    # This makes learning a lot easier...
    def terminate_early(self, objects, obs):
        # Check if the object is clear
        block_name = objects[0].name
        # Could recompute, but am lazy
        pred_obs = get_observation(obs).literals
        if clear(block_name) not in pred_obs and \
            holding(block_name) not in pred_obs:
            return True
        # If we're holding something else, terminate
        for lit in pred_obs:
            if lit.predicate == holding and lit.variables[0] != block_name:
                return True
        return False

    def move_to_above_block(objects, obs):
        relative_grasp_position = np.array([0., 0., 0.])
        block_name = objects[0].name
        gripper_position, _, left_finger_pos = obs['gripper']
        block_position = obs['blocks'][block_name][3:6]
        target_position = np.add(block_position, relative_grasp_position)
        target_position[2] += 0.15
        if (gripper_position[0] - target_position[0])**2 + (gripper_position[1] - target_position[1])**2 < atol:
            return np.zeros(4), True
        return get_move_action(gripper_position, target_position, atol=atol), False

    def open_grippers(objects, obs):
        return np.array([0., 0., 0., 1.]), True

    def move_down(objects, obs):
        relative_grasp_position = np.array([0., 0., 0.])
        block_name = objects[0].name
        gripper_position, _, left_finger_pos = obs['gripper']
        block_position = obs['blocks'][block_name][3:6]
        target_position = np.add(block_position, relative_grasp_position) 
        done = ((gripper_position[2] - target_position[2])**2 < atol)
        return np.array([0., 0., -0.25, 0.]), done

    def close_grippers(objects, obs):
        return np.array([0., 0., 0., -0.25]), True

    def bring_to_pick_position(objects, obs):
        pick_height = 0.5
        relative_grasp_position = np.array([0., 0., 0.])
        block_name = objects[0].name
        gripper_position, _, left_finger_pos = obs['gripper']
        block_position = obs['blocks'][block_name][3:6]
        target_position = np.add(block_position, relative_grasp_position)    
        done = (gripper_position[2] > pick_height)
        return np.array([0., 0., 0.5, -0.005]), done

    def pause(objects, obs):
        return np.array([0., 0., 0.5, -0.005]), True

    stages = [
        move_to_above_block,
        open_grippers,
        move_down,
        close_grippers,
        close_grippers,
        bring_to_pick_position,
        pause,
        pause,
        pause,
        pause,
        pause
    ]


class PutonController(StateMachineController):
    # This makes learning a lot easier...
    def terminate_early(self, objects, obs):
        # Check if the object is clear
        block_name = objects[0].name
        # Could recompute, but am lazy
        if clear(block_name) not in get_observation(obs).literals:
            return True
        return False

    def move_to_above_block(objects, obs):
        block_name = objects[0].name
        gripper_position, _, left_finger_pos = obs['gripper']
        block_position = obs['blocks'][block_name][3:6]
        target_position = block_position.copy()
        target_position[2] += 0.15
        if np.sum(np.subtract(target_position, gripper_position)**2) < atol:
            return np.array([0., 0., 0., -0.005]), True
        move_action = get_move_action(gripper_position, target_position, atol=atol)
        move_action = np.array(move_action)
        move_action[3] = -0.005
        return move_action, False

    def open_grippers(objects, obs):
        return np.array([0., 0., 0., 1.]), True

    def move_up(objects, obs):
        return np.array([0., 0., 1., 0.]), True

    stages = [
        move_to_above_block,
        open_grippers,
        move_up,
        move_up,
        move_up,
        move_up,
        move_up,
    ]

class PutontableController(StateMachineController):
    stages = []
    open_position = None

    # This makes learning a lot easier...
    def terminate_early(self, objects, obs):
        # Could recompute, but am lazy
        if handempty() in get_observation(obs).literals:
            return True
        return False

    def reset(self):
        PutontableController.open_position = None
        super().reset()

    def find_open_position(obs):
        x = 1.25
        z = 0.15
        min_y, max_y = 0.5, 0.5 + 0.4
        block_ys = []
        for block_state in obs['blocks'].values():
            # Only look at blocks on the table
            if 0.1 < block_state[5] < 0.3:
                block_ys.append(block_state[4])
        best_y = min_y
        best_y_dist = 0
        for y in np.linspace(min_y, max_y, num=20):
            y_safe = True
            y_dist = np.inf
            for block_y in block_ys:
                y_dist = min(y_dist, abs(block_y - y))
            if y_dist > best_y_dist:
                best_y_dist = y_dist
                best_y = y
        return np.array([x, best_y, z])

    def move_to_above_open_position(objects, obs):

        gripper_position, _, left_finger_pos = obs['gripper']

        if PutontableController.open_position is None:
            PutontableController.open_position = PutontableController.find_open_position(obs)

        target_position = PutontableController.open_position.copy()
        target_position[2] += 0.15
        if np.sum(np.subtract(target_position, gripper_position)**2) < atol:
            return np.array([0., 0., 0., -0.005]), True
        move_action = get_move_action(gripper_position, target_position, atol=atol)
        move_action = np.array(move_action)
        move_action[3] = -0.005
        return move_action, False

    def open_grippers(objects, obs):
        return np.array([0., 0., 0., 1.]), True

    def move_up(objects, obs):
        return np.array([0., 0., 1., 0.]), True

    stages = [
        move_to_above_open_position,
        open_grippers,
        move_up,
        move_up,
        move_up,
        move_up,
        move_up,
    ]

controllers = {
    pickup : PickupController(),
    puton : PutonController(),
    putontable : PutontableController(),
}


# Noise effect
noiseoutcome = Predicate("noiseoutcome", 0, [])

# State predicates
on = Predicate("on", 2, [block_type, block_type])
ontable = Predicate("ontable", 1, [block_type])
holding = Predicate("holding", 1, [block_type])
clear = Predicate("clear", 1, [block_type])
handempty = Predicate("handempty", 0, [])
observation_predicates = [on, ontable, holding, clear, handempty, noiseoutcome]


def get_observation(state):
    # First check whether we're holding a block
    holding_block = None

    gripper_position, _, _ = state['gripper']
    for block_name in state["blocks"]:
        block_position = state['blocks'][block_name][3:6]
        if np.sum(np.subtract(gripper_position, block_position)**2) < 1e-3:
            holding_block = block_name
            break

    # Find which blocks are vertically aligned, indicating a pile,
    piles = []

    for block_name, block_state in state["blocks"].items():
        if block_name == holding_block:
            continue
        x, y, z = block_state[3:6]
        contender_pile = None
        best_pile_dist = np.inf
        for pile in piles:
            for block_in_pile in pile:
                base_x, base_y = state["blocks"][block_in_pile][3:5]
                contender_pile_dist = abs(x - base_x) + abs(y - base_y)
                if contender_pile_dist < 1e-1 and contender_pile_dist < best_pile_dist:
                    contender_pile = pile
                    best_pile_dist = contender_pile_dist
        if contender_pile is None:
            piles.append([block_name])
        else:
            contender_pile.append(block_name)
            # sort in increasing height order
            contender_pile.sort(key=lambda s : state["blocks"][s][5])

    # Build predicates
    obs = set()

    if holding_block is not None:
        obs.add(holding(holding_block))
    else:
        obs.add(handempty())

    for pile in piles:
        obs.add(ontable(pile[0]))
        obs.add(clear(pile[-1]))
        if len(pile) > 1:
            for below, above in zip(pile[:-1], pile[1:]):
                obs.add(on(above, below))

    # Extract objects
    all_objects = {o for lit in obs for o in lit.variables}

    # goal set in superclass
    state = State(frozenset(obs), frozenset(all_objects), None)

    return state

# TODO move this somewhere else, it is general
def create_abstract_pybullet_env(low_level_cls, controllers, get_observation, obs_preds,
                                 controller_max_steps=100):

    class AbstractPybulletEnv(gym.Env):
        action_predicates = list(controllers.keys())
        observation_predicates = obs_preds

        def __init__(self, record_low_level_video=False, video_out=None, *args, **kwargs):
            self.action_space = LiteralSpace(self.action_predicates)
            self.observation_space = LiteralSetSpace(set(self.observation_predicates))
            self.low_level_env = low_level_cls(*args, **kwargs)

            if record_low_level_video:
                self.low_level_env = VideoWrapper(self.low_level_env, video_out)

        def reset(self):
            low_level_obs, debug_info = self.low_level_env.reset()
            obs = get_observation(low_level_obs)
            self._previous_low_level_obs = low_level_obs
            return obs, debug_info

        def step(self, action):
            controller = controllers[action.predicate]
            controller.reset()
            low_level_obs = self._previous_low_level_obs
            reward = 0.
            done = False
            for _ in range(controller_max_steps):
                control, controller_done = controller.step(action.variables, low_level_obs)
                if controller_done:
                    break
                low_level_obs, low_level_reward, done, debug_info = self.low_level_env.step(control)
                reward += low_level_reward
                if done:
                    break
            obs = get_observation(low_level_obs)
            self._previous_low_level_obs = low_level_obs 
            return obs, reward, done, {}

        def render(self, *args, **kwargs):
            return self.low_level_env.render(*args, **kwargs)

        def close(self):
            return self.low_level_env.close()

        def seed(self, seed=None):
            return self.low_level_env.seed(seed=seed)

    return AbstractPybulletEnv

BasePybulletBlocksEnv = create_abstract_pybullet_env(LowLevelPybulletBlocksEnv, controllers, 
    get_observation, observation_predicates)


class PybulletBlocksEnv(BasePybulletBlocksEnv):
    """Add some high-level goals
    """

    def fix_problem_index(self, idx):
        
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Dummy variables for the source code.
        self.problems = [0]
        self.problem_idx = 0

        self._goals = [
            LiteralConjunction([on("block2", "block1"), on("block1", "block0")]),
            LiteralConjunction([on("block1", "block2"), on("block2", "block0")]),
            LiteralConjunction([on("block1", "block2"), ontable("block0")]),
        ]
        self._num_goals = len(self._goals)
        self._goal = None

    def reset(self):
        while True:
            self._goal = self._goals[self.low_level_env.np_random.choice(self._num_goals)]
            obs, debug_info = super().reset()
            if self._goal.holds(obs.literals):
                continue
            obs = obs.with_goal(self._goal)
            debug_info["problem_file"] = 0
            return obs, debug_info

    def step(self, action):
        obs, _, _, debug_info = super().step(action)
        reward = 1.0 if self._goal.holds(obs.literals) else 0.0
        done = reward == 1.
        obs = obs.with_goal(self._goal)
        #print("Done: ", done)
        debug_info["problem_file"] = 0
        return obs, reward, done, debug_info


