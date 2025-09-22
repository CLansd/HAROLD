"""
python scripts/skrl/train.py --task=Harold-Walking --num_envs 4 --headless --max_iterations 10
python scripts/skrl/train.py --task=Harold-Walking --num_envs 4096 --headless --max_iterations 450
python scripts/skrl/play.py --task=Harold-Walking --num_envs 1 --device cpu

2392MB when running 1 environment.
3569MB when running 256 environments.
4759MB when running 512 environments.
7131MB when running 1024 environments.

tensorboard --logdir logs
localhost:6006
"""

### --- PARAMETERS --- ###

# GAIT
gait_period             =   0.55                # Time taken for the robot to take one step (seconds).
gait_height             =   0.10                # Maximum height of feet above the ground during a step (meters).

# VEL COMMANDS
# The BRAVER paper used an initial range of (-0.3,0.5) fwd/back and (-0.4,0.4) in left/right,
# and increased fwd/back up to (-1.2,2.5) by the end of their curriculum.
lin_vel_x_min           =  -0.0                 # Lower bound of commanded x-direction velocities (meters/second).
lin_vel_x_max           =   0.0                 # Upper bound of commanded x-direction velocities (meters/second).
lin_vel_y_min           =  -0.0                 # Lower bound of commanded y-direction velocities (meters/second).
lin_vel_y_max           =   0.0                 # Upper bound of commanded y-direction velocities (meters/second).
ang_vel_z_min           =  -0.0                 # Lower bounnd of commanded z-direction angular velocities (radians/second).
ang_vel_z_max           =   0.0                 # Upper bound of commanded z-direction angular velocities (radians/second).
vel_resampling_period   =   10.0                # Time between resamplings of the velocity commands (seconds).
fraction_still          =   0.02                # Sampled probability of all environments which should stand still.

# ACTIONS
# The BRAVER paper used a value of 0.25.
joint_action_scale      =   0.25                # Scale factor applied to the agent network's output before sending to actuators.

# OBSERVATIONS
# History values default to 0.
# If history is set to x, the agent observes the current value as well as the past x values.
joint_pos_error_history =   4                   # 4 per the BRAVER paper.
joint_vel_history       =   5                   # 5 per the BRAVER paper.
actions_history         =   4                   # 4 per the BRAVER paper.

# REWARDS
xy_lin_vel_rew_weight   =   1.0                 # Reward weight for accurately tracking the xy velocity.
z_ang_vel_rew_weight    =   0.5                 # Reward weight for accurately tracking the z axis angular velocity.
z_lin_vel_rew_weight    =  -2.0                 # Reward weight for accurately maintaining a z velocity of zero.
xy_ang_vel_rew_weight   =  -0.015               # Reward weight for accurately maintaining an xy angular velocity of zero.
feet_tracking_weight    =  -16.0                # Reward weight for accurately tracking feet with their reference trajectories.
flat_body_weight        =  -0.5                 # Reward weight for keeping the body close to vertical.

# SIMULATION
# Observations and actions are recomputed every (decimation_factor * physics_time_step) seconds,
# whereas the scene's physics are recomputed every physics_time_step seconds.
# A frame is rendered every (render_interval_factor * physics_time_step) seconds.
decimation_factor       =   4                   # Decimation factor.
physics_time_step       =   0.005               # Length of physics time step (seconds).
render_interval_factor  =   4                   # Render interval factor.
episode_length          =   20.0                # Episode length (seconds).
camera_pos              =   (2.0, 2.0, 1.2)     # Position of the camera/viewport in the scene (meters).

# ARTICULATION INITIALIZATION
root_init_pos           =   (0.0, 0.0, 0.40)    # Initial position of the articulation root in world frame (meters).
joints_init_pos         =   {                   # Initial positions of the joints (radians).
    "LeftHipJoint": 0.0,
    "RightHipJoint": 0.0,
    "LeftThighJoint": -0.72,
    "RightThighJoint": 0.72,
    "LeftCalfJoint": 1.39626,
    "RightCalfJoint": -1.39626,
}
joint_init_vels         =   {".*": 0.0}         # Initial velocities of the joints (radians/second).
soft_joint_lim_factor   =   0.9                 # Soft joint position limit factor.

# ACTUATORS
# 22.0Nm is the maximum for GIM8108-8.
actuator_max_torque     =   22.0                # Maximum actuator torque (Newton - meters).
# 320 rpm (33.5 rad/s) is the maximum for GIM8108-8.
actuator_ang_vel_limit  =   33.5                # Maximum actuator angular velocity (radians/second).
# Gain from Kayden's paper, multiplied by 1.5 since our legs are about 1.5 times longer.
actuator_stiffness      =   15.0                # Actuator proportional gain.
# Gain from Kayden's paper, multiplied by 1.5 since our legs are about 1.5 times longer.
actuator_damping        =   0.45                # Actuator damping gain.