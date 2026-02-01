### --- PARAMETERS --- ###

# GAIT
gait_resampling_period  =   5.0                 # Time between resamplings of the gait command (seconds).
gait_freq_min           =   1.5                 # Minimum gait frequency (Hz).
gait_freq_max           =   2.5                 # Maximum gait frequency (Hz).
gait_phase_offs_min     =   0.5                 # Minimum gait phase offset [0-1].
gait_phase_offs_max     =   0.5                 # Maximum gait phase offset [0-1].
gait_durations_min      =   0.5                 # Contact durations range minimum [0-1].
gait_durations_max      =   0.5                 # Contact durations range maximum [0-1].
gait_swing_height_min   =   0.1                 # Foot swing height min (meters).
gait_swing_height_max   =   0.2                 # Foot swing height max (meters).

# VEL COMMANDS
# The BRAVER paper used an initial range of (-0.3,0.5) fwd/back and (-0.4,0.4) in left/right,
# and increased fwd/back up to (-1.2,2.5) by the end of their curriculum.
lin_vel_x_min           =  -0.5                 # Lower bound of commanded x-direction velocities (meters/second).
lin_vel_x_max           =   0.5                 # Upper bound of commanded x-direction velocities (meters/second).
lin_vel_y_min           =  -0.5                 # Lower bound of commanded y-direction velocities (meters/second).
lin_vel_y_max           =   0.5                 # Upper bound of commanded y-direction velocities (meters/second).
ang_vel_z_min           =  -0.5                 # Lower bounnd of commanded z-direction angular velocities (radians/second).
ang_vel_z_max           =   0.5                 # Upper bound of commanded z-direction angular velocities (radians/second).
vel_resamp_per_min      =   0.0                 # Minimum time between resamplings of the velocity command (seconds).
vel_resamp_per_max      =   5.0                 # Maximum time between resamplings of the velocity command (seconds).
fraction_still          =   0.02                # Sampled probability of all environments which should stand still.

# ACTIONS
# The BRAVER paper used a value of 0.25.
joint_action_scale      =   0.25                # Scale factor applied to the agent network's output before sending to actuators.

# OBSERVATIONS
# History values default to 0.
# If history is set to x, the agent observes the current value as well as the past x values.
obs_history_length      =   10                  # History length of observations.

# EVENTS
static_friction_min     =   0.4                 # Minimum coefficient of static friction for random distribution.
static_friction_max     =   1.2                 # Maximum coefficient of static friction for random distribution.
dynamic_friction_min    =   0.7                 # Minimum coefficient of dynamic friction for random distribution.
dynamic_friction_max    =   0.9                 # Maximum coefficient of dynamic friction for random distribution.
restitution_min         =   0.0                 # Minimum coefficient of restitution for random distribution.
restitution_max         =   1.0                 # Maximum coefficient of restitution for random distribution.
x_pose_range_min        =  -0.5                 # Minimum sampled x coordinate on reset.
x_pose_range_max        =   0.5                 # Maximum sampled x coordinate on reset.
y_pose_range_min        =  -0.5                 # Minimum sampled y coordinate on reset.
y_pose_range_max        =   0.5                 # Maximum sampled y coordinate on reset.
yaw_pose_range_min      =  -3.14                # Minimum sampled yaw on reset.
yaw_pose_range_max      =   3.14                # Maximum sampled yaw on reset.
reset_vel_x_min         =  -0.5                 # Minimum sampled x velocity on reset.
reset_vel_x_max         =   0.5                 # Maximum sampled x velocity on reset.
reset_vel_y_min         =  -0.5                 # Minimum sampled y velocity on reset.
reset_vel_y_max         =   0.5                 # Maximum sampled y velocity on reset.
reset_vel_z_min         =  -0.5                 # Minimum sampled z velocity on reset.
reset_vel_z_max         =   0.5                 # Maximum sampled z velocity on reset.
reset_roll_vel_min      =  -0.5                 # Minimum sampled roll angular velocity on reset.
reset_roll_vel_max      =   0.5                 # Maximum sampled roll angular velocity on reset.
reset_pitch_vel_min     =  -0.5                 # Minimum sampled pitch angular velocity on reset.
reset_pitch_vel_max     =   0.5                 # Maximum sampled pitch angular velocity on reset.
reset_yaw_vel_min       =  -0.5                 # Minimum sampled yaw angular velocity on reset.
reset_yaw_vel_max       =   0.5                 # Maximum sampled yaw angular velocity on reset.

# REWARDS
keep_balance_weight     =   1.0                 # Reward weight for staying alive at each time step.
rew_lin_vel_xy_weight   =   1.0                 # Reward weight for accurately tracking the xy velocity.
rew_ang_vel_z_weight    =   0.5                 # Reward weight for accurately tracking the z axis angular velocity.
target_height           =   0.35                # Target height to keep the robot's body at.
pen_base_height_weight  =  -20.0                # Reward weight for keeping base height close to desired.
pen_lin_vel_z_weight    =  -0.5                 # Reward weight for accurately maintaining a z velocity of zero.
pen_ang_vel_xy_weight   =  -0.05                # Reward weight for accurately maintaining an xy angular velocity of zero.
pen_joint_torque_weight =  -0.00008             # Penalizes joint torques.
pen_joint_accel_weight  =  -2.5e-07             # Penalizes joint accelerations.
pen_action_rate_weight  =  -0.03                # Penalizes the action rate.
pen_actn_smooth_weight  =  -0.04                # Penalizes action smoothness?
flat_body_weight        =  -10.0                # Reward weight for keeping the body close to vertical.
pen_joint_vel_l2_weight =  -1e-03               # Penalizes joint velocity l2 norm.
pen_joint_powers_weight =  -5e-04               # Penalizes joint powers.


feet_tracking_weight    =  -20.0                # Reward weight for accurately tracking feet with their reference trajectories.
termination_penalty     =  -0.0                 # Termination penalty.

# SIMULATION
# Observations and actions are recomputed every (decimation_factor * physics_time_step) seconds,
# whereas the scene's physics are recomputed every physics_time_step seconds.
# A frame is rendered every (render_interval_factor * physics_time_step) seconds.
decimation_factor       =   4                   # Decimation factor.
physics_time_step       =   0.005               # Length of physics time step (seconds).
render_interval_factor  =   4                   # Render interval factor.
episode_length          =   20.0                # Episode length (seconds).
camera_pos              =   (8.0, 8.0, 4.8)     # Position of the camera/viewport in the scene (meters).

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
actuator_max_torque     =   300.0               # Maximum actuator torque (Newton - meters).
# 320 rpm (33.5 rad/s) is the maximum for GIM8108-8.
actuator_ang_vel_limit  =   300.0               # Maximum actuator angular velocity (radians/second).
# Gain from Kayden's paper, multiplied by 1.5 is 15.0 since our legs are about 1.5 times longer.
actuator_stiffness      =   15.0                # Actuator proportional gain.
# Gain from Kayden's paper, multiplied by 1.5 is 0.45 since our legs are about 1.5 times longer.
actuator_damping        =   0.90                # Actuator damping gain.