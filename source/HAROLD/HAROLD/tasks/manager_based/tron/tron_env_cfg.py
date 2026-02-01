### --- IMPORTS --- ###
import math
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from .tron import HAROLD_CFG
from . import tron_cfg
from . import mdp

### --- SCENE DEFINITION --- ###
@configclass
class HaroldSceneCfg(InteractiveSceneCfg):

    # Ground Plane.
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0,100.0)),
    )

    # Robot.
    robot: ArticulationCfg = HAROLD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Sensors.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=4, track_air_time=True, update_period=0.0
    )

    # Lights.
    light = AssetBaseCfg(
        prim_path="/World/SkyLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=750.0,
        ),
    )

### --- MDP COMMANDS --- ###
@configclass
class CommandsCfg:
    # The commanded base linear and angular velocity setpoints.
    base_velocity   = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(tron_cfg.vel_resamp_per_min,tron_cfg.vel_resamp_per_max),
        rel_standing_envs=tron_cfg.fraction_still,
        heading_command=False, # Whether to use the heading command or angular velocity command.
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(tron_cfg.lin_vel_x_min,tron_cfg.lin_vel_x_max),
            lin_vel_y=(tron_cfg.lin_vel_y_min,tron_cfg.lin_vel_y_max),
            ang_vel_z=(tron_cfg.ang_vel_z_min,tron_cfg.ang_vel_z_max),
        ),
    )

    gait_command = mdp.UniformGaitCommandCfg(
        resampling_time_range=(tron_cfg.gait_resampling_period,tron_cfg.gait_resampling_period),
        debug_vis=False,
        ranges=mdp.UniformGaitCommandCfg.Ranges(
            frequencies=(tron_cfg.gait_freq_min,tron_cfg.gait_freq_max),
            offsets=(tron_cfg.gait_phase_offs_min,tron_cfg.gait_phase_offs_max),
            durations=(tron_cfg.gait_durations_min, tron_cfg.gait_durations_max),
            swing_height=(tron_cfg.gait_swing_height_min, tron_cfg.gait_swing_height_max)
        )
    )

### --- MDP ACTIONS --- ###
@configclass
class ActionsCfg:
    joint_effort = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint", "knee_L_Joint", "knee_R_Joint"],
        scale=tron_cfg.joint_action_scale,
    )

### --- MDP OBSERVATIONS --- ###
@configclass
class ObservationsCfg:

    # Define the observation terms available to the agent.
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel                = ObsTerm(func=mdp.base_ang_vel, history_length=tron_cfg.obs_history_length, noise=GaussianNoise(mean=0.0, std=0.05),clip=(-100.0, 100.0),scale=0.25)
        proj_gravity                = ObsTerm(func=mdp.projected_gravity, history_length=tron_cfg.obs_history_length, noise=GaussianNoise(mean=0.0, std=0.025),clip=(-100.0, 100.0),scale=1.0)
        joint_pos                   = ObsTerm(func=mdp.joint_pos_rel, history_length=tron_cfg.obs_history_length, noise=GaussianNoise(mean=0.0, std=0.01),clip=(-100.0, 100.0),scale=1.0)
        joint_vel                   = ObsTerm(func=mdp.joint_vel, history_length=tron_cfg.obs_history_length, noise=GaussianNoise(mean=0.0, std=0.01),clip=(-100.0, 100.0),scale=0.05)
        last_action                 = ObsTerm(func=mdp.last_action, history_length=tron_cfg.obs_history_length)
        velocity_command            = ObsTerm(
            func=mdp.generated_commands,
            history_length=tron_cfg.obs_history_length,
            params={
                "command_name": "base_velocity",
            }
        )
        gait_phase = ObsTerm(func=mdp.get_gait_phase, history_length=tron_cfg.obs_history_length)
        gait_command                = ObsTerm(
            func=mdp.get_gait_command,
            history_length=tron_cfg.obs_history_length,
            params={
                "command_name": "gait_command"
            }
        )

        # Post initialization.
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # LimX used a separate encoder (and thus a separate ObsGroup) for history -> we should consider doing this too in the future.
        # But for now we will just use the built-in history_length option for simplicity.

    @configclass
    class CriticCfg(ObsGroup):
        # LimX didn't add history to any of these terms, but I'm going to add them and see if it works ok.
        # Policy Observations
        base_ang_vel                = ObsTerm(func=mdp.base_ang_vel, history_length=tron_cfg.obs_history_length)
        proj_gravity                = ObsTerm(func=mdp.projected_gravity, history_length=tron_cfg.obs_history_length)
        joint_pos                   = ObsTerm(func=mdp.joint_pos_rel, history_length=tron_cfg.obs_history_length)
        joint_vel                   = ObsTerm(func=mdp.joint_vel, history_length=tron_cfg.obs_history_length)
        last_action                 = ObsTerm(func=mdp.last_action, history_length=tron_cfg.obs_history_length)
        velocity_command            = ObsTerm(
            func=mdp.generated_commands,
            history_length=tron_cfg.obs_history_length,
            params={
                "command_name": "base_velocity",
            }
        )
        gait_phase = ObsTerm(func=mdp.get_gait_phase)
        gait_command                = ObsTerm(
            func=mdp.get_gait_command,
            history_length=tron_cfg.obs_history_length,
            params={
                "command_name": "gait_command"
            }
        )

        # Privileged Observations
        base_lin_vel                = ObsTerm(func=mdp.base_lin_vel, history_length=tron_cfg.obs_history_length)
        height                      = ObsTerm(func=mdp.base_pos_z, history_length=tron_cfg.obs_history_length)  # LimX uses a height scanner sensor for this, but for now I think this should be OK since we are on flat terrain.
        robot_joint_torque          = ObsTerm(func=mdp.robot_joint_torque, history_length=tron_cfg.obs_history_length)
        robot_joint_acc             = ObsTerm(func=mdp.robot_joint_acc, history_length=tron_cfg.obs_history_length)
        robot_feet_contact_force = ObsTerm(
            func=mdp.robot_feet_contact_force,
            history_length=tron_cfg.obs_history_length,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot_[LR]_Link"),
            },
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_pos = ObsTerm(func=mdp.robot_pos, history_length=tron_cfg.obs_history_length)
        robot_vel = ObsTerm(func=mdp.robot_vel, history_length=tron_cfg.obs_history_length)
        robot_material_properties = ObsTerm(func=mdp.robot_material_properties)
        robot_base_pose = ObsTerm(func=mdp.robot_base_pose)

        # Post initialization.
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

### --- MDP EVENTS --- ###
@configclass
class EventCfg:
    # ON STARTUP
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_[LR]_Link"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    radomize_rigid_body_mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.7, 0.9),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 48,
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (32, 48),
            "damping_distribution_params": (2.0, 3.0),
            "operation": "abs",
            "distribution": "uniform",
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    robot_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_coms,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_distribution_params": ((-0.075, 0.075), (-0.05, 0.06), (-0.05, 0.05)),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # ON RESET:
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-0.5, 0.5),
            "velocity_range": (0.0, 0.0),
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )

    # interval
    push_robot = EventTerm(
        func=mdp.apply_external_force_torque_stochastic,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            "force_range": {
                "x": (-500.0, 500.0),
                "y": (-500.0, 500.0),
                "z": (-0.0, 0.0),
            },  # force = mass * dv / dt
            "torque_range": {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (-0.0, 0.0)},
            "probability": 0.002,  # Expect step = 1 / probability
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    

### --- MDP REWARDS --- ###
@configclass
class RewardsCfg:
    keep_balance                    = RewTerm(
        func=mdp.stay_alive,
        weight=tron_cfg.keep_balance_weight
    )
    rew_lin_vel_xy                  = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=tron_cfg.rew_lin_vel_xy_weight,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.2)
        },
    )
    rew_ang_vel_z                   = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=tron_cfg.rew_ang_vel_z_weight,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.2)
        },
    )
    pen_base_height = RewTerm(
        func=mdp.base_com_height,
        params={
            "target_height": 0.65,
        },
        weight=-20.0,
    )
    pen_lin_vel_z                   = RewTerm(func=mdp.lin_vel_z_l2, weight=tron_cfg.pen_lin_vel_z_weight)
    pen_ang_vel_xy                  = RewTerm(func=mdp.ang_vel_xy_l2, weight=tron_cfg.pen_ang_vel_xy_weight)
    pen_joint_torque                = RewTerm(func=mdp.joint_torques_l2, weight=tron_cfg.pen_joint_torque_weight)
    pen_joint_accel                 = RewTerm(func=mdp.joint_acc_l2, weight=tron_cfg.pen_joint_accel_weight)
    pen_action_rate                 = RewTerm(func=mdp.action_rate_l2, weight=tron_cfg.pen_action_rate_weight)
    pen_action_smoothness           = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=tron_cfg.pen_actn_smooth_weight)
    pen_flat_orientation            = RewTerm(func=mdp.flat_orientation_l2, weight=tron_cfg.flat_body_weight)
    # pen_feet_distance not included until we switch to a point foot CAD model.
    # pen_feet_regulation not included until we switch to a point foot CAD model.
    # foot_landing_vel not included until we switch to a point foot CAD model.
    pen_joint_vel_l2                = RewTerm(func=mdp.joint_vel_l2, weight=tron_cfg.pen_joint_vel_l2_weight)
    pen_joint_powers                = RewTerm(func=mdp.joint_powers_l1, weight=tron_cfg.pen_joint_powers_weight)
    test_gait_reward = RewTerm(
        func=mdp.GaitReward,
        weight=1.0,
        params={
            "tracking_contacts_shaped_force": -2.0,
            "tracking_contacts_shaped_vel": -2.0,
            "gait_force_sigma": 25.0,
            "gait_vel_sigma": 0.25,
            "kappa_gait_probs": 0.05,
            "command_name": "gait_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
        },
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


### --- MDP TERMINATIONS --- ###
@configclass
class TerminationsCfg:
    time_out        = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "base_Link",
                    "abad_L_Link",  # Corrected
                    "abad_R_Link",  # Corrected
                    "hip_L_Link",   # Corrected
                    "hip_R_Link",   # Corrected
                    "knee_L_Link",  # Corrected
                    "knee_R_Link",  # Corrected
                ],
            ),
            "threshold": 1.0,
        },
    )
    

### --- ENVIRONMENT CONFIGURATION --- ###
@configclass
class HaroldEnvCfg(ManagerBasedRLEnvCfg):
    scene: HaroldSceneCfg = HaroldSceneCfg(num_envs=4096, env_spacing=2.5) # Default num_envs and env_spacing.
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        self.decimation = tron_cfg.decimation_factor
        self.episode_length_s = tron_cfg.episode_length
        self.viewer.eye = tron_cfg.camera_pos
        self.sim.dt = tron_cfg.physics_time_step
        self.sim.render_interval = tron_cfg.render_interval_factor
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt