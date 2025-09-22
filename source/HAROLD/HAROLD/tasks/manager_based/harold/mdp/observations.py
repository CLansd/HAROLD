import torch
import math
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor import ContactSensor
from isaaclab.sensors import FrameTransformer
import isaaclab.utils.math as math_utils


# I've validated that this function seems to be working correctly.
def feet_contact(
    env: ManagerBasedRLEnv, sensor_cfg_L: SceneEntityCfg, sensor_cfg_R: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor_L: ContactSensor = env.scene.sensors[sensor_cfg_L.name]
    contact_sensor_R: ContactSensor = env.scene.sensors[sensor_cfg_R.name]

    contact_threshold = 1.0

    # Shape is (N,B,3) where N is the number of sensors and B is the number of bodies in each sensor.
    # In this case, I think that means it is (1,1,3).
    left_forces = contact_sensor_L.data.net_forces_w
    right_forces = contact_sensor_R.data.net_forces_w

    contacts_L = torch.norm(left_forces, dim=-1).max(dim=-1)[0] > contact_threshold
    contacts_R = torch.norm(right_forces, dim=-1).max(dim=-1)[0] > contact_threshold

    feet_contacts = torch.stack(
        [contacts_L.float(), contacts_R.float()], dim=-1
    )  # Now has dimension of (1,2).
    return feet_contacts


def phase_tensor(env: ManagerBasedRLEnv, T: float) -> torch.Tensor:
    # env.sim.time was not correct, as it is not yet initialized upon the initial observation function call causing an error.
    current_time = env.episode_length_buf.float() * env.physics_dt * env.cfg.decimation

    # Use torch operations for vectorized calculations
    half_T = 0.5 * T
    t1 = torch.remainder(current_time, T)
    t2 = torch.remainder(current_time + half_T, T)

    # Calculate phi values using torch.where instead of if/else
    # torch.where(condition, value_if_true, value_if_false)
    phi1 = torch.where(t1 < half_T, t1 / half_T, (t1 - half_T) / half_T)
    phi2 = torch.where(t2 < half_T, t2 / half_T, (t2 - half_T) / half_T)

    # Calculate l values using torch.where
    l1 = torch.where(t1 < half_T, 0.0, 1.0)
    l2 = torch.where(t2 < half_T, 0.0, 1.0)

    # Calculate sin/cos components
    sin_1 = torch.sin((phi1 + l1) * math.pi)
    cos_1 = torch.cos((phi1 + l1) * math.pi)
    sin_2 = torch.sin((phi2 + l2) * math.pi)
    cos_2 = torch.cos((phi2 + l2) * math.pi)

    # Stack the results into a final tensor. Shape: (num_envs, 4)
    phase_obs = torch.stack([sin_1, cos_1, sin_2, cos_2], dim=-1)

    return phase_obs


# I think this is probably working after some basic print statements and dropping the robot from a height.
def joint_pos_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_pos_target = asset.data.joint_pos_target
    joint_pos = asset.data.joint_pos
    joint_pos_err = joint_pos - joint_pos_target

    return joint_pos_err


# I think I've verified that this works.
def root_euler_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation as euler angles (roll, pitch, yaw) in the environment frame.

    Returns:
        torch.Tensor: The root orientation as euler angles (roll, pitch, yaw) of shape (num_envs, 3).
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # get the root quaternion in the world frame: (w, x, y, z)
    quat_w = asset.data.root_quat_w
    # convert quaternion to euler angles (roll, pitch, yaw)
    # The result is in radians.

    # This utility returns a tuple of tensors (roll, pitch, yaw)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_w)
    # We stack them into a single tensor of shape (num_envs, 3)
    euler_ang = torch.stack((roll, pitch, yaw), dim=-1)

    return euler_ang


def stance_phase_pos(Lx: torch.Tensor, Ly: torch.Tensor, T: float, t: torch.Tensor) -> torch.Tensor:
    # Vectorized calculation for x and y positions
    x_pos = (Lx / 2) - (2 * t * Lx) / T
    y_pos = (Ly / 2) - (2 * t * Ly) / T
    # Z position is a tensor of zeros with the same shape as the inputs
    z_pos = torch.zeros_like(Lx)
    
    # Stack the results into a (num_envs, 3) tensor
    return torch.stack([x_pos, y_pos, z_pos], dim=-1)


def swing_phase_pos(Lx: torch.Tensor, Ly: torch.Tensor, h_max: float, T: float, t: torch.Tensor) -> torch.Tensor:
    # Common sinusoidal term
    sin_term = torch.sin((4 * math.pi * t) / T)
    cos_term = torch.cos((4 * math.pi * t) / T)

    # Vectorized calculation for x, y, and z positions
    x_pos = (-Lx / 2) + (Lx * (((2 * t) / T) - (1 / (2 * math.pi)) * sin_term))
    y_pos = (-Ly / 2) + (Ly * (((2 * t) / T) - (1 / (2 * math.pi)) * sin_term))
    z_pos = (h_max / 2) * (1 - cos_term)
    
    # Stack the results into a (num_envs, 3) tensor
    return torch.stack([x_pos, y_pos, z_pos], dim=-1)

# I've verified that this function does in fact seem to work correctly using 3D plots of its outputs.
def reference_feet_positions(env: ManagerBasedRLEnv, command_name: str, T: float, h_max: float) -> torch.Tensor:
    """Calculates the reference positions for the left and right feet based on a walking gait pattern."""
    # These are tensors of shape (num_envs,)
    current_time = env.episode_length_buf.float() * env.physics_dt * env.cfg.decimation
    command = env.command_manager.get_command(command_name)
    vx = command[:, 0]
    vy = command[:, 1]
    
    # Lx and Ly are also tensors of shape (num_envs,)
    Lx = (T * vx) / 2
    Ly = (T * vy) / 2

    # tL and tR are the primary clocks for the left and right feet
    tL = current_time % T
    tR = (current_time + (T / 2)) % T
    
    # Create boolean masks for each of the four quadrants of the gait cycle based on the left foot's clock
    mask_q1 = (tL < T / 4)
    mask_q2 = (tL >= T / 4) & (tL < T / 2)
    mask_q3 = (tL >= T / 2) & (tL < (3 * T) / 4)
    # mask_q4 is the remaining 'else' case

    # -- Calculate the potential foot positions for EACH of the 4 cases --

    # Case 1: tL in [0, T/4) -> Left: stance, Right: swing
    pos_L_c1 = stance_phase_pos(Lx, Ly, T, tL + T / 4)
    pos_R_c1 = swing_phase_pos(Lx, Ly, h_max, T, tR - T / 4)

    # Case 2: tL in [T/4, T/2) -> Left: swing, Right: stance
    pos_L_c2 = swing_phase_pos(Lx, Ly, h_max, T, tL - T / 4)
    pos_R_c2 = stance_phase_pos(Lx, Ly, T, tR - (3 * T) / 4)

    # Case 3: tL in [T/2, 3T/4) -> Left: swing, Right: stance
    pos_L_c3 = swing_phase_pos(Lx, Ly, h_max, T, tL - T / 4)
    pos_R_c3 = stance_phase_pos(Lx, Ly, T, tR + T / 4)

    # Case 4: tL in [3T/4, T) -> Left: stance, Right: swing
    pos_L_c4 = stance_phase_pos(Lx, Ly, T, tL - (3 * T) / 4)
    pos_R_c4 = swing_phase_pos(Lx, Ly, h_max, T, tR - T / 4)

    # -- Select the correct positions using the masks --
    # Unsqueeze masks from (num_envs,) to (num_envs, 1) to allow broadcasting
    mask_q1 = mask_q1.unsqueeze(-1)
    mask_q2 = mask_q2.unsqueeze(-1)
    mask_q3 = mask_q3.unsqueeze(-1)
    
    # Use nested `where` clauses to pick the result from the correct case
    foot_pos_L = torch.where(mask_q1, pos_L_c1,
                             torch.where(mask_q2, pos_L_c2,
                                         torch.where(mask_q3, pos_L_c3, pos_L_c4)))
    
    foot_pos_R = torch.where(mask_q1, pos_R_c1,
                             torch.where(mask_q2, pos_R_c2,
                                         torch.where(mask_q3, pos_R_c3, pos_R_c4)))
    
    # -- Apply the offsets to the foot positions --
    # Define the offset vectors as tensors on the correct device
    device = foot_pos_L.device
    offset_L = torch.tensor([-0.0898, -0.0533, -0.3441], device=device)
    offset_R = torch.tensor([0.0898, -0.0533, -0.3441], device=device)

    # Add the offsets to the calculated foot positions
    foot_pos_L += offset_L
    foot_pos_R += offset_R

    # Concatenate the final foot positions into a single flat tensor
    feet_positions = torch.cat([foot_pos_L, foot_pos_R], dim=-1)

    return feet_positions


# In observations.py

def feet_pos_in_root_frame(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Extracts the feet positions in the root frame from a FrameTransformer sensor.

    This function assumes that a FrameTransformer sensor has been configured in the
    environment's scene to track the left and right foot frames relative to the
    robot's base frame. The sensor's data is read directly.

    Returns:
        torch.Tensor: The feet positions in the root frame, shaped (num_envs, 6).
                      The order is [left_x, left_y, left_z, right_x, right_y, right_z].
    """
    # Access the frame transformer sensor from the scene using its configured name.
    frame_sensor: FrameTransformer = env.scene[sensor_cfg.name]

    # Shape: (num_envs, 2, 3) -> [x, y, z] position for each foot.
    feet_pos_data = frame_sensor.data.target_pos_source
    # Shape: (num_envs, 2, 4) -> [w, x, y, z] quaternion for each foot.
    feet_quat_data = frame_sensor.data.target_quat_source

    # 2. Define the offsets in the local frame of each foot.
    # Assumes the sensor is configured with the left foot first, then the right.
    # Shape: (2, 3)
    offsets_local = torch.tensor(
        [
            [-0.09, -0.08, -0.448],  # Left foot local offset
            [0.09, -0.08, -0.448],   # Right foot local offset
        ],
        device=env.device,
        dtype=torch.float32,
    )

    # -- FIX IS HERE --
    # Expand offsets_local to have the same batch dimension as feet_quat_data.
    # Shape goes from (2, 3) -> (1, 2, 3) -> (num_envs, 2, 3).
    # This is a memory-efficient operation that just repeats the tensor metadata.
    num_envs = env.num_envs
    offsets_local_expanded = offsets_local.unsqueeze(0).expand(num_envs, -1, -1)
    # -- END OF FIX --

    # 3. Rotate the local offsets by the foot's current orientation.
    # This transforms the offset vector from the local foot frame to the root frame.
    # `quat_apply` now receives tensors of compatible shapes:
    # quat: (num_envs, 2, 4) and vec: (num_envs, 2, 3)
    offsets_in_root_frame = math_utils.quat_apply(feet_quat_data, offsets_local_expanded)

    # 4. Add the rotated offset to the original foot position.
    offset_foot_positions = feet_pos_data + offsets_in_root_frame

    # Reshape to a flat observation vector (num_envs, 6) for the policy.
    flattened_feet_pos_data = offset_foot_positions.view(env.num_envs, -1)
    return flattened_feet_pos_data