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
from .leg import LEG_CFG
from .. import mdp

### --- SCENE DEFINITION --- ###
@configclass
class LegSceneCfg(InteractiveSceneCfg):

    # Robot.
    robot: ArticulationCfg = LEG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Lights.
    light = AssetBaseCfg(
        prim_path="/World/SkyLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=750.0,
        ),
    )

### --- MDP ACTIONS --- ###
@configclass
class ActionsCfg:
    joint_effort = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Hip", "Thigh_01", "Knee"],
        scale=0.25,
    )

### --- MDP OBSERVATIONS --- ###
@configclass
class ObservationsCfg:

    # Define the observation terms available to the agent.
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos                   = ObsTerm(func=mdp.joint_pos_rel, history_length=10)
        joint_vel                   = ObsTerm(func=mdp.joint_vel, history_length=10)
        phase_signal = ObsTerm(func=mdp.phase_sin_cos, params={"T": 2.0})

        # Post initialization.
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        joint_pos                   = ObsTerm(func=mdp.joint_pos_rel, history_length=10)
        joint_vel                   = ObsTerm(func=mdp.joint_vel, history_length=10)
        phase_signal = ObsTerm(func=mdp.phase_sin_cos, params={"T": 2.0})

        robot_joint_torque          = ObsTerm(func=mdp.robot_joint_torque, history_length=10)
        robot_joint_acc             = ObsTerm(func=mdp.robot_joint_acc, history_length=10)

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

    # ON RESET:
    
    # THIS IS ABSOLUTELY CRUCIAL TO GETTING THE ROBOT TO NOT YEET ITSELF INTO THE AIR WHEN reset_root_state_uniform is enabled!!!
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Hip", "Thigh_01", "Knee"]),
            "position_range": (1.0, 1.0), # I think we will need to adjust this.
            "velocity_range": (0.0, 0.0),
        },
        is_global_time=False,
        min_step_count_between_reset=0,
    )
    

### --- MDP REWARDS --- ###
@configclass
class RewardsCfg:
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0)
    trajectory_tracking = RewTerm(
        func=mdp.track_triangular_trajectory_world,
        weight=-4.0,
        params={
            # "command_name": "base_velocity",  <-- REMOVED
            "asset_cfg": SceneEntityCfg("robot", body_names=["Calf"]),
            "step_height": 0.1,
            "step_length": 0.2, # <-- ADDED (Total stride length in meters)
        },
    )


### --- MDP TERMINATIONS --- ###
@configclass
class TerminationsCfg:
    time_out        = DoneTerm(func=mdp.time_out, time_out=True)
    

### --- ENVIRONMENT CONFIGURATION --- ###
@configclass
class LegEnvCfg(ManagerBasedRLEnvCfg):
    scene: LegSceneCfg = LegSceneCfg(num_envs=4096, env_spacing=2.5) # Default num_envs and env_spacing.
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.viewer.eye = (8.0, 8.0, 4.8)
        self.sim.dt = 0.005
        self.sim.render_interval = 4
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15