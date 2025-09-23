# rewards.py

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


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
