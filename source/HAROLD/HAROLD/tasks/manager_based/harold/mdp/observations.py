import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor import ContactSensor
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