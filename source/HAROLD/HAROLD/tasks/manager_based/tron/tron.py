import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from . import tron_cfg

### --- ARTICULATION DEFINITION --- ###
HAROLD_CFG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path=f"/home/c/Documents/harold/HAROLD/PF_TRON1A/PF_TRON1A.usd", # Path to USD file.
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
        pos=(0.0, 0.0, 0.84),
        joint_pos={
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=tron_cfg.soft_joint_lim_factor,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "abad_L_Joint",
                "abad_R_Joint",
                "hip_L_Joint",
                "hip_R_Joint",
                "knee_L_Joint",
                "knee_R_Joint",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "abad_L_Joint": 40.0,
                "abad_R_Joint": 40.0,
                "hip_L_Joint": 40.0,
                "hip_R_Joint": 40.0,
                "knee_L_Joint": 40.0,
                "knee_R_Joint": 40.0,
            },
            damping={
                "abad_L_Joint": 2.5,
                "abad_R_Joint": 2.5,
                "hip_L_Joint": 2.5,
                "hip_R_Joint": 2.5,
                "knee_L_Joint": 2.5,
                "knee_R_Joint": 2.5,
            },
        ),
    },
)

"""
    actuators={
        "LeftHip": ImplicitActuatorCfg(
            joint_names_expr=["LeftHipJoint"],
            effort_limit_sim=tron_cfg.actuator_max_torque,
            velocity_limit_sim=tron_cfg.actuator_ang_vel_limit,
            stiffness={
                ".*": tron_cfg.actuator_stiffness
            },
            damping={
                ".*": tron_cfg.actuator_damping
            },
        ),
        "RightHip": ImplicitActuatorCfg(
            joint_names_expr=["RightHipJoint"],
            effort_limit_sim=tron_cfg.actuator_max_torque,
            velocity_limit_sim=tron_cfg.actuator_ang_vel_limit,
            stiffness={
                ".*": tron_cfg.actuator_stiffness
            },
            damping={
                ".*": tron_cfg.actuator_damping
            },
        ),
        "LeftThigh": ImplicitActuatorCfg(
            joint_names_expr=["LeftThighJoint"],
            effort_limit_sim=tron_cfg.actuator_max_torque,
            velocity_limit_sim=tron_cfg.actuator_ang_vel_limit,
            stiffness={
                ".*": tron_cfg.actuator_stiffness
            },
            damping={
                ".*": tron_cfg.actuator_damping
            },
        ),
        "RightThigh": ImplicitActuatorCfg(
            joint_names_expr=["RightThighJoint"],
            effort_limit_sim=tron_cfg.actuator_max_torque,
            velocity_limit_sim=tron_cfg.actuator_ang_vel_limit,
            stiffness={
                ".*": tron_cfg.actuator_stiffness
            },
            damping={
                ".*": tron_cfg.actuator_damping
            },
        ),
        "LeftCalf": ImplicitActuatorCfg(
            joint_names_expr=["LeftCalfJoint"],
            effort_limit_sim=tron_cfg.actuator_max_torque,
            velocity_limit_sim=tron_cfg.actuator_ang_vel_limit,
            stiffness={
                ".*": tron_cfg.actuator_stiffness
            },
            damping={
                ".*": tron_cfg.actuator_damping
            },
        ),
        "RightCalf": ImplicitActuatorCfg(
            joint_names_expr=["RightCalfJoint"],
            effort_limit_sim=tron_cfg.actuator_max_torque,
            velocity_limit_sim=tron_cfg.actuator_ang_vel_limit,
            stiffness={
                ".*": tron_cfg.actuator_stiffness
            },
            damping={
                ".*": tron_cfg.actuator_damping
            },
        ),
    },
"""