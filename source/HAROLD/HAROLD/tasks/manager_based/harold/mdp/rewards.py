# rewards.py

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.sensors import FrameTransformer
from isaaclab.assets import Articulation
import isaaclab.utils.math as math_utils
from .observations import reference_feet_positions, feet_pos_in_root_frame

def feet_position_tracking_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    T: float,
    h_max: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Calculates the unscaled L2 distance penalty between reference and actual foot positions.
    The final reward is scaled by the 'weight' parameter in the environment configuration.
    """
    # 1. Get the reference and actual foot positions.
    ref_pos = reference_feet_positions(env, command_name, T, h_max)
    actual_pos = feet_pos_in_root_frame(env, sensor_cfg)

    # 2. Reshape the tensors for easier computation.
    ref_pos_reshaped = ref_pos.view(env.num_envs, 2, 3)
    actual_pos_reshaped = actual_pos.view(env.num_envs, 2, 3)

    # 3. Calculate the L2-norm (Euclidean distance) for each foot.
    distance_error = torch.norm(ref_pos_reshaped - actual_pos_reshaped, p=2, dim=-1)

    # 4. Sum the errors from both feet to get a single error value per environment.
    total_error = torch.sum(distance_error, dim=-1)

    # 5. Return the raw penalty. The RewardManager will apply the negative weight.
    # Since your weight is negative, this term should be positive.
    return total_error


def air_time_reward(
        env: ManagerBasedRLEnv,
        left_sensor_cfg: SceneEntityCfg,
        right_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    left_contact_sensor: ContactSensor = env.scene.sensors[left_sensor_cfg.name]
    right_contact_sensor: ContactSensor = env.scene.sensors[right_sensor_cfg.name]
    # Note: We don't want to use last air time since it will keep rewarding the robot even if it only takes one step!!!
    # 1) Get the current air time for the left calf.
    left_current_air_time = left_contact_sensor.data.current_air_time
    # 2) Get the current air time for the right calf.
    right_current_air_time = right_contact_sensor.data.current_air_time
    # 3) Sum the left and right air times.
    summed_air_times = left_current_air_time + right_current_air_time
    summed_air_times = summed_air_times.squeeze(-1)
    # 4) Return the summed value (in seconds), which will then be scaled by the RewardsTerm API separately.
    return summed_air_times


def calc_phi_i(t: torch.Tensor, T: float) -> torch.Tensor:
    half_T = T / 2.0
    phi_i = torch.where(t < half_T, t / half_T, (t - half_T) / half_T)
    return phi_i


def feet_speed_rew(env: "ManagerBasedRLEnv", T: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Computes the linear velocity of the robot's feet in the world frame."""

    asset = env.scene[asset_cfg.name]

    left_calf_idx = asset.find_bodies("LeftCalf")[0][0]
    right_calf_idx = asset.find_bodies("RightCalf")[0][0]

    # This quantity is the linear velocity of the articulation bodies' center of mass frame.
    left_calf_vel_w = asset.data.body_lin_vel_w[:, left_calf_idx, :]
    right_calf_vel_w = asset.data.body_lin_vel_w[:, right_calf_idx, :]

    # The angular velocity of the articulation bodies in the world frame.
    left_calf_ang_vel_w = asset.data.body_ang_vel_w[:, left_calf_idx, :]
    right_calf_ang_vel_w = asset.data.body_ang_vel_w[:, right_calf_idx, :]

    # The pose of the articulation bodies in the world frame.
    left_calf_pose_w = asset.data.body_pose_w[:, left_calf_idx, :]
    right_calf_pose_w = asset.data.body_pose_w[:, right_calf_idx, :]
    # Extract quaternions for rotating from local to world frame.
    left_calf_quat_w = left_calf_pose_w[:, 3:7]
    right_calf_quat_w = right_calf_pose_w[:, 3:7]

    # Center of mass positions of all bodies in their respective link frames.
    left_calf_com_pos_b = asset.data.body_com_pos_b[:, left_calf_idx, :]
    right_calf_com_pos_b = asset.data.body_com_pos_b[:, right_calf_idx, :]

    # Foot offsets in the local frame of the LeftCalf and Right Calf.
    offsets_local = torch.tensor(
        [
            [-0.09, -0.08, -0.448],  # Left foot local offset in the frame of the LeftCalf.
            [0.09, -0.08, -0.448],   # Right foot local offset in the frame of the RightCalf.
        ],
        device=env.device,
        dtype=torch.float32,
    )
    # Unsqueeze to allow broadcasting with the batched calf data.
    left_foot_offset_b = offsets_local[0].unsqueeze(0)
    right_foot_offset_b = offsets_local[1].unsqueeze(0)

    # -- Start of New Code --

    # 1. Calculate the vector from the Center of Mass (CoM) to the foot in the local body frame.
    # This is r_foot/CoM = r_foot/origin - r_CoM/origin
    r_left_foot_from_com_b = left_foot_offset_b - left_calf_com_pos_b
    r_right_foot_from_com_b = right_foot_offset_b - right_calf_com_pos_b

    # 2. Rotate these vectors from the local body frame into the world frame using the calf's orientation.
    r_left_foot_from_com_w = math_utils.quat_apply(left_calf_quat_w, r_left_foot_from_com_b)
    r_right_foot_from_com_w = math_utils.quat_apply(right_calf_quat_w, r_right_foot_from_com_b)

    # 3. Calculate the tangential velocity component (omega x r) for each foot.
    # This represents the additional linear velocity at the foot's position due to the calf's rotation.
    left_vel_from_rot = torch.cross(left_calf_ang_vel_w, r_left_foot_from_com_w, dim=-1)
    right_vel_from_rot = torch.cross(right_calf_ang_vel_w, r_right_foot_from_com_w, dim=-1)

    # 4. Compute the total velocity of each foot in the world frame.
    # v_foot = v_CoM + (omega x r_foot/CoM)
    left_foot_vel_w = left_calf_vel_w + left_vel_from_rot
    right_foot_vel_w = right_calf_vel_w + right_vel_from_rot

    current_time = env.episode_length_buf.float() * env.physics_dt * env.cfg.decimation

    # tL and tR are the primary clocks for the left and right feet
    tL = current_time % T
    tR = (current_time + (T / 2.0)) % T

    phi_L = calc_phi_i(tL, T)
    phi_R = calc_phi_i(tR, T)

    right_foot_vel_norm = torch.norm(right_foot_vel_w, p=2, dim=-1)

    # Case 1: tL in [0, T/4) -> Left: stance, Right: swing
    # Case 2: tL in [T/4, T/2) -> Left: swing, Right: stance
    # Case 3: tL in [T/2, 3T/4) -> Left: swing, Right: stance
    # Case 4: tL in [3T/4, T) -> Left: stance, Right: swing

    is_left_stance = (tL < (T / 4.0)) | (tL >= (3.0 * T / 4.0))

    left_penalty = torch.norm(left_foot_vel_w * torch.sin(phi_L * torch.pi).unsqueeze(-1), dim=-1)
    right_penalty = torch.norm(right_foot_vel_w * torch.sin(phi_R * torch.pi).unsqueeze(-1), dim=-1)

    reward_term = torch.where(is_left_stance, left_penalty, right_penalty)

    return reward_term


def feet_contact_during_swing_rew(env: "ManagerBasedRLEnv", sensor_cfg_L: SceneEntityCfg, sensor_cfg_R: SceneEntityCfg, T: float) -> torch.Tensor:
    contact_sensor_L: ContactSensor = env.scene.sensors[sensor_cfg_L.name]
    contact_sensor_R: ContactSensor = env.scene.sensors[sensor_cfg_R.name]

    current_time = env.episode_length_buf.float() * env.physics_dt * env.cfg.decimation

    # tL and tR are the primary clocks for the left and right feet
    tL = current_time % T
    tR = (current_time + (T / 2.0)) % T

    phi_L = calc_phi_i(tL, T)
    phi_R = calc_phi_i(tR, T)

    # Shape is (N,B,3) where N is the number of sensors and B is the number of bodies in each sensor.
    # In this case, I think that means it is (1,1,3).
    left_forces = contact_sensor_L.data.net_forces_w
    right_forces = contact_sensor_R.data.net_forces_w

    left_penalty = torch.norm(left_forces * torch.sin((phi_L * torch.pi) + torch.pi).unsqueeze(-1).unsqueeze(-1), dim=-1)
    right_penalty = torch.norm(right_forces * torch.sin((phi_R * torch.pi) + torch.pi).unsqueeze(-1).unsqueeze(-1), dim=-1)

    # Case 1: tL in [0, T/4) -> Left: stance, Right: swing
    # Case 2: tL in [T/4, T/2) -> Left: swing, Right: stance
    # Case 3: tL in [T/2, 3T/4) -> Left: swing, Right: stance
    # Case 4: tL in [3T/4, T) -> Left: stance, Right: swing

    is_left_stance = (tL < (T / 4.0)) | (tL >= (3.0 * T / 4.0))

    reward_term = torch.where(is_left_stance.unsqueeze(-1), right_penalty, left_penalty)

    return reward_term.squeeze(-1)

