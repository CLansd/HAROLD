import isaaclab.sim as sim_utils
import os
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from robot_lab import ROBOT_LAB_EXT_DIR

### --- ARTICULATION DEFINITION --- ###
LEG_CFG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path=f"{ROBOT_LAB_EXT_DIR}/source/robot_lab/assets/USD_LEG_R2/USD_LEG_R2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={                 # Initial positions of the joints (radians).
            "Hip": 0.0,
            "Thigh_01": 0.0,
            "Knee": 0.0,
            },
        joint_vel={".*": 0.0},      # Initial velocities of the joints (radians/second).
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "HipActuator": ImplicitActuatorCfg(
            joint_names_expr=["Hip"],
            effort_limit_sim = 60.0,
            velocity_limit_sim=19.9,
            stiffness={
                ".*": 60.0
            },
            damping={
                ".*": 4.0
            },
        ),
        "ThighActuator": ImplicitActuatorCfg(
            joint_names_expr=["Thigh_01"],
            effort_limit_sim=60.0,
            velocity_limit_sim=19.9,
            stiffness={
                ".*": 60.0
            },
            damping={
                ".*": 4.0
            },
        ),
        "KneeActuator": ImplicitActuatorCfg(
            joint_names_expr=["Knee"],
            effort_limit_sim=60.0,
            velocity_limit_sim=19.9,
            stiffness={
                ".*": 60.0
            },
            damping={
                ".*": 4.0
            },
        ),
    },
)