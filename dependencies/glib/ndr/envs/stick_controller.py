import numpy as np

DEBUG = False

def get_move_action(gripper_position, target_position, atol=1e-3, gain=10., close_gripper=False):
    """
    Move an end effector to a position and orientation.
    """
    # Get the currents
    action = gain * np.subtract(target_position, gripper_position)
    if close_gripper:
        gripper_action = -0.1
    else:
        gripper_action = 0.
    action = np.hstack((action, gripper_action))

    return action

def block_is_grasped(left_finger_pos, gripper_position, block_position, relative_grasp_position, atol=1e-3):
    block_inside = block_inside_grippers(gripper_position, block_position, relative_grasp_position, atol=atol)
    # print("block_inside?", block_inside)
    grippers_closed = grippers_are_closed(left_finger_pos, atol=atol)
    # print('grippers_closed?', grippers_closed)

    return block_inside and grippers_closed

def block_inside_grippers(gripper_position, block_position, relative_grasp_position, atol=1e-3):
    relative_position = np.subtract(gripper_position, block_position)

    return np.sum(np.subtract(relative_position, relative_grasp_position)**2) < atol

def grippers_are_closed(left_finger_pos, atol=1e-3):
    return abs(left_finger_pos) - 0.024 <= atol

def grippers_are_open(left_finger_pos, atol=1e-3):
    return abs(left_finger_pos - 0.05) <= atol


def pick_at_position(left_finger_pos, gripper_position, block_position, place_position, relative_grasp_position=(0., 0., -0.02), workspace_height=0.1, atol=1e-3):
    """
    Returns
    -------
    action : [float] * 4
    """

    # If the gripper is already grasping the block
    if block_is_grasped(left_finger_pos, gripper_position, block_position, relative_grasp_position, atol=atol):

        # If the block is already at the place position, do nothing except keep the gripper closed
        if np.sum(np.subtract(block_position, place_position)**2) < atol:
            if DEBUG:
                print("The block is already at the place position; do nothing")
            return np.array([0., 0., 0., -1.])

        # Move to the place position while keeping the gripper closed
        target_position = np.add(place_position, relative_grasp_position)
        target_position[2] += workspace_height/2.
        if DEBUG:
            print("Move to above the place position")
        return get_move_action(gripper_position, target_position, atol=atol, close_gripper=True)

    # If the block is ready to be grasped
    if block_inside_grippers(gripper_position, block_position, relative_grasp_position, atol=atol):

        # Close the grippers
        if DEBUG:
            print("Close the grippers")
        return np.array([0., 0., 0., -1.])

    # If the gripper is above the block
    target_position = np.add(block_position, relative_grasp_position)    
    if (gripper_position[0] - target_position[0])**2 + (gripper_position[1] - target_position[1])**2 < atol:

        # If the grippers are closed, open them
        if not grippers_are_open(left_finger_pos, atol=atol):
            if DEBUG:
                print("Open the grippers")
            return np.array([0., 0., 0., 1.])

        # Move down to grasp
        if DEBUG:
            print("Move down to grasp")
        return get_move_action(gripper_position, target_position, atol=atol)


    # Else move the gripper to above the block
    target_position[2] += workspace_height
    if DEBUG:
        print("Move to above the block")
    return get_move_action(gripper_position, target_position, atol=atol)


def get_stick_control(obs, origin, atol=1e-2):
    """
    Returns
    -------
    action : [float] * 4
    """
    # relative_grasp_position = np.array([0., 0., -0.03])
    # stick_hold_height = -0.25 #0.6
    # stick_slide_height = -0.3 #0.5
    # stick_grasp_offset = -0.42 #-0.45

    relative_grasp_position = np.array([0., 0., -0.04])
    stick_hold_height = origin[2] + 0.25 #-0.25 #0.6
    stick_slide_height = origin[2] + 0.2 #-0.3 #0.5
    stick_grasp_offset = -0.45

    gripper_position, left_finger_pos, stick_position, block_position, block_velocity = np.split(obs['observation'], [3, 4, 7, 10])
    place_position = obs['desired_goal']

    # Done
    if abs(block_position[0] - place_position[0]) + abs(block_position[1] - place_position[1]) <= atol:
        if DEBUG:
            print("DONE")
        return np.array([0., 0., 0., -0.01])

    if block_is_grasped(left_finger_pos, gripper_position, stick_position, relative_grasp_position=relative_grasp_position, atol=atol):
        
        # Stick is beyond the block
        stick_target = np.array([block_position[0] + stick_grasp_offset, block_position[1], stick_hold_height])
        
        horizontal_align = abs(stick_position[1] - stick_target[1]) < atol
        vertical_align = stick_position[0] - stick_grasp_offset >= stick_target[0] + atol
        if horizontal_align and vertical_align:

            # Stick is down, so sweep
            if abs(gripper_position[2] - stick_slide_height) <= atol:

                if DEBUG:
                    print("Sweeping back")

                direction = np.subtract(place_position, block_position[:2])
                direction = direction / np.linalg.norm(direction)

                return np.array([0.5 * direction[0], 0.5 * direction[1], 0.0, -0.1])

            # Move stick down
            stick_target[2] = stick_slide_height

            if DEBUG:
                print("Putting the stick down")
            # return np.array([0., 0., -0.4, -0.1])
            return get_move_action(gripper_position, stick_target, close_gripper=True)

        # Stick is high enough
        if stick_position[2] >= stick_hold_height - atol:

            # Bring the stick just beyond the block
            stick_target = np.array([block_position[0] + stick_grasp_offset, block_position[1], stick_hold_height])

            if DEBUG:
                print("Bringing the stick to beyond the block", gripper_position[:2], stick_target[:2])
            return get_move_action(gripper_position, stick_target, close_gripper=True)

        # Pick up the stick
        stick_target = np.array([gripper_position[0], gripper_position[1], stick_hold_height])

        if DEBUG:
            print("Picking up the stick", horizontal_align, vertical_align)
        # return get_move_action(gripper_position, stick_target, close_gripper=True)
        return np.array([0., 0., 1., -0.1])



    # Grasp the stick
    if DEBUG:
        print("Grasping and lifting the stick")
    stick_target = stick_position.copy()
    stick_target[2] = stick_hold_height
    return pick_at_position(left_finger_pos, gripper_position, stick_position, stick_target, relative_grasp_position=relative_grasp_position)






# def OLD_get_stick_control(obs, atol=1e-2):
#     """
#     Returns
#     -------
#     action : [float] * 4
#     """
#     # gripper_position = obs['observation'][:3]
#     # block_position = obs['observation'][3:6]
#     # left_finger_pos = obs['observation'][9]
#     # stick_position = obs['observation'][25:28]
#     # place_position = obs['desired_goal']

#     relative_grasp_position = np.array([0., 0., -0.03])
#     stick_hold_height = -0.2 #0.6
#     stick_slide_height = -0.3 #0.5
#     stick_grasp_offset = -0.5 #-0.45

#     gripper_position, left_finger_pos, stick_position, block_position = np.split(obs['observation'], [3, 4, 7])
#     place_position = obs['desired_goal']

#     # Done
#     if abs(block_position[0] - place_position[0]) + abs(block_position[1] - place_position[1]) <= atol:
#         if DEBUG:
#             print("DONE")
#         return np.array([0., 0., 0., -1.])

#     # Grasp and lift the stick
#     if not block_is_grasped(left_finger_pos, gripper_position, stick_position, relative_grasp_position=relative_grasp_position, atol=atol):
#         if DEBUG:
#             print("Grasping and lifting the stick")
#         stick_target = stick_position.copy()
#         stick_target[2] = stick_hold_height
#         return pick_at_position(left_finger_pos, gripper_position, stick_position, stick_target, relative_grasp_position=relative_grasp_position)

#     # Pick up the stick
#     stick_target = np.array([stick_position[0], stick_position[1], stick_hold_height])

#     if abs(stick_target[2] - stick_position[2]) > atol:
#         if DEBUG:
#             print("Picking up the stick")
#         return get_move_action(gripper_position, stick_target, close_gripper=True)

#     # Bring the stick just beyond the block
#     stick_target = np.array([block_position[0] + stick_grasp_offset, block_position[1], stick_hold_height])

#     if abs(stick_position[1] - stick_target[1]) > atol or (stick_position[0] - stick_grasp_offset/2. < stick_target[0] + atol):
#         if DEBUG:
#             print("Bringing the stick to beyond the block", stick_position[:2], stick_target[:2])
#         return get_move_action(gripper_position, stick_target, close_gripper=True)

#     # Putting the stick down
#     stick_target = np.array([stick_position[0], stick_position[1], stick_slide_height])

#     if abs(gripper_position[2] - stick_target[2]) > atol:
#         if DEBUG:
#             print("Putting the stick down")
#         return np.array([0., 0., -0.4, -1.])


#     if DEBUG:
#         print("Sweeping back")

#     direction = np.subtract(place_position, block_position)
#     direction = direction[:2] / np.linalg.norm(direction[:2])

#     return np.array([0.4 * direction[0], 0.4 * direction[1], 0., -1.])






