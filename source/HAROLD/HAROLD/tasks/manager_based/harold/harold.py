import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from . import harold_cfg

### --- ARTICULATION DEFINITION --- ###
HAROLD_CFG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path=f"/home/c/Documents/harold/HAROLD/USD_HAROLD_R1/HAROLD_R1_FLAT.usd", # Path to USD file.
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
        pos=harold_cfg.root_init_pos,
        joint_pos=harold_cfg.joints_init_pos,
        joint_vel=harold_cfg.joint_init_vels,
    ),
    soft_joint_pos_limit_factor=harold_cfg.soft_joint_lim_factor,
    actuators={
        "LeftHip": ImplicitActuatorCfg(
            joint_names_expr=["LeftHipJoint"],
            effort_limit_sim=harold_cfg.actuator_max_torque,
            velocity_limit_sim=harold_cfg.actuator_ang_vel_limit,
            stiffness={
                ".*": harold_cfg.actuator_stiffness
            },
            damping={
                ".*": harold_cfg.actuator_damping
            },
        ),
        "RightHip": ImplicitActuatorCfg(
            joint_names_expr=["RightHipJoint"],
            effort_limit_sim=harold_cfg.actuator_max_torque,
            velocity_limit_sim=harold_cfg.actuator_ang_vel_limit,
            stiffness={
                ".*": harold_cfg.actuator_stiffness
            },
            damping={
                ".*": harold_cfg.actuator_damping
            },
        ),
        "LeftThigh": ImplicitActuatorCfg(
            joint_names_expr=["LeftThighJoint"],
            effort_limit_sim=harold_cfg.actuator_max_torque,
            velocity_limit_sim=harold_cfg.actuator_ang_vel_limit,
            stiffness={
                ".*": harold_cfg.actuator_stiffness
            },
            damping={
                ".*": harold_cfg.actuator_damping
            },
        ),
        "RightThigh": ImplicitActuatorCfg(
            joint_names_expr=["RightThighJoint"],
            effort_limit_sim=harold_cfg.actuator_max_torque,
            velocity_limit_sim=harold_cfg.actuator_ang_vel_limit,
            stiffness={
                ".*": harold_cfg.actuator_stiffness
            },
            damping={
                ".*": harold_cfg.actuator_damping
            },
        ),
        "LeftCalf": ImplicitActuatorCfg(
            joint_names_expr=["LeftCalfJoint"],
            effort_limit_sim=harold_cfg.actuator_max_torque,
            velocity_limit_sim=harold_cfg.actuator_ang_vel_limit,
            stiffness={
                ".*": harold_cfg.actuator_stiffness
            },
            damping={
                ".*": harold_cfg.actuator_damping
            },
        ),
        "RightCalf": ImplicitActuatorCfg(
            joint_names_expr=["RightCalfJoint"],
            effort_limit_sim=harold_cfg.actuator_max_torque,
            velocity_limit_sim=harold_cfg.actuator_ang_vel_limit,
            stiffness={
                ".*": harold_cfg.actuator_stiffness
            },
            damping={
                ".*": harold_cfg.actuator_damping
            },
        ),
    },
)

"""
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=harold_cfg.actuator_max_torque,
            velocity_limit_sim=harold_cfg.actuator_ang_vel_limit,
            stiffness={
                ".*": harold_cfg.actuator_stiffness
            },
            damping={
                ".*": harold_cfg.actuator_damping
            },
        ),
    },
"""