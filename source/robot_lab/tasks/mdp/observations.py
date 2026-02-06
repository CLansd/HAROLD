import torch
import math
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor import ContactSensor
from isaaclab.assets import Articulation
import isaaclab.utils.math as math_utils

def phase_sin_cos(env: ManagerBasedRLEnv, T: float) -> torch.Tensor:
    """
    Observes the sin and cos of the current gait phase.
    This provides a continuous clock signal to the agent.
    
    Args:
        T: The period of the walking cycle in seconds.
    """
    # Calculate continuous phase [0, 1]
    current_time = env.episode_length_buf * env.step_dt
    freq = 1.0 / T
    phase = (current_time * freq) % 1.0
    
    # Convert to Sin/Cos to avoid discontinuity at 0/1
    # Shape: (num_envs, 2)
    phase_signal = torch.stack([
        torch.sin(2 * math.pi * phase),
        torch.cos(2 * math.pi * phase)
    ], dim=-1)
    
    return phase_signal

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

def get_gait_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Get the current gait command parameters as observation.

    Returns:
        torch.Tensor: The gait command parameters [frequency, offset, duration].
                     Shape: (num_envs, 3).
    """
    return env.command_manager.get_command(command_name)

def get_gait_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the current gait phase as observation.

    The gait phase is represented by [sin(phase), cos(phase)] to ensure continuity.
    The phase is calculated based on the episode length and gait frequency.

    Returns:
        torch.Tensor: The gait phase observation. Shape: (num_envs, 2).
    """
    # check if episode_length_buf is available
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, 2, device=env.device)

    # Get the gait command from command manager
    command_term = env.command_manager.get_term("gait_command")
    # Calculate gait indices based on episode length
    gait_indices = torch.remainder(env.episode_length_buf * env.step_dt * command_term.command[:, 0], 1.0)
    # Reshape gait_indices to (num_envs, 1)
    gait_indices = gait_indices.unsqueeze(-1)
    # Convert to sin/cos representation
    sin_phase = torch.sin(2 * torch.pi * gait_indices)
    cos_phase = torch.cos(2 * torch.pi * gait_indices)

    return torch.cat([sin_phase, cos_phase], dim=-1)


def robot_feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """contact force of the robot feet"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    contact_force_tensor = contact_sensor.data.net_forces_w_history.to(device)
    return contact_force_tensor.view(contact_force_tensor.shape[0], -1)

def robot_joint_torque(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint torque of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return_val = asset.data.applied_torque.to(device)
    torque_values = return_val[0].tolist()
    formatted_torques = [f"{torque:8.2f}" for torque in torque_values]
    output_string = "[" + ", ".join(formatted_torques) + "]"
    # Order is: ["LeftHipJoint", "RightHipJoint", "LeftThighJoint", "RightThighJoint", "LeftCalfJoint", "RightCalfJoint"]
    # print(output_string)
    return return_val

def robot_joint_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint acc of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.joint_acc.to(device)

def robot_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_mass.to(device)


def robot_inertia(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """inertia of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    inertia_tensor = asset.data.default_inertia.to(device)
    return inertia_tensor.view(inertia_tensor.shape[0], -1)

def robot_joint_stiffness(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint stiffness of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_stiffness.to(device)


def robot_joint_damping(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint damping of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_damping.to(device)

def robot_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """pose of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w.to(device)


def robot_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """velocity of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_vel_w.to(device)

def robot_material_properties(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """material properties of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    material_tensor = asset.root_physx_view.get_material_properties().to(device)
    return material_tensor.view(material_tensor.shape[0], -1)

def robot_base_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """pose of the robot base"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w.to(device)