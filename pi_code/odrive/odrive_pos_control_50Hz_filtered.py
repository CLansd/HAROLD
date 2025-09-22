import signal
import can
import sys
import time
import struct
import matplotlib.pyplot as plt
from collections import deque

####################################################
# --- TO DO ---
# - Should a velocity feedforward term be added into our script?

####################################################
# --- CONFIG PARAMETERS ---

# --- Step Function Parameters ---
SETPOINT_1_TIME = 2.0 # Seconds spent in setpoint 1.
SETPOINT_2_TIME = 2.0 # Seconds spent in setpoint 2.
STEP_SIZE = 0.2 # Step function step size (turns).

# --- Loop Timing ---
CONTROL_LOOP_RATE_HZ = 50 # Control loop frequency.


# --- CAN Configuration ---
CAN_INTERFACE = 'socketcan'
CHANNEL = 'can0'
ODRIVE_AXIS_ID = 4 # Match ODrive's configured Node ID for Axis 0.





####################################################
# --- CONSTANTS ---

# --- ODrive CAN Command IDs ---
CMD_ID_SET_AXIS_REQUESTED_STATE = 0x007
CMD_ID_SET_CONTROLLER_MODES = 0x00B
CMD_ID_SET_INPUT_POS = 0x00C
CMD_ID_GET_ENCODER_ESTIMATES = 0x009 # Used to receive Pos/Vel estimates

# --- ODrive Axis States ---
AXIS_STATE_IDLE = 1
AXIS_STATE_CLOSED_LOOP_CONTROL = 8

# --- ODrive Control Modes ---
CONTROL_MODE_POSITION_CONTROL = 3

# --- ODrive Input Modes ---
INPUT_MODE_PASSTHROUGH = 1 # Directly set PID controller setpoint with no filtering or trajectory smoothing.
INPUT_MODE_POS_FILTER = 3





####################################################
# --- GLOBALS ---
bus = None
initialized = False
running = True
last_position_turns = 0.0
last_velocity_turns_per_sec = 0.0
loop_time = 0.0





####################################################
# --- HELPER FUNCTIONS ---

# --- Helper function to pack and create CAN message ---
def odrive_msg(axis_id, cmd_id, data=None, data_fmt=""):
    arbitration_id = (axis_id << 5) | cmd_id
    packed_data = b''
    if data is not None and data_fmt:
        try:
            if not isinstance(data, tuple): data = (data,)
            packed_data = struct.pack(data_fmt, *data)
        except struct.error as e:
            print(f"Error packing data for cmd {cmd_id:#03x}: {e}", file=sys.stderr)
            return None
    return can.Message(
        arbitration_id=arbitration_id, data=packed_data, is_extended_id=False, dlc=len(packed_data)
    )

# --- Signal handler for graceful shutdown ---
def signal_handler(sig, frame):
    global running
    print("\nCtrl+C detected. Requesting script exit...")
    running = False

signal.signal(signal.SIGINT, signal_handler)





####################################################
# --- INITIALIZATION ---

# --- Plotting Setup ---
max_data_points = int((SETPOINT_1_TIME + SETPOINT_2_TIME) * CONTROL_LOOP_RATE_HZ)
time_data = deque(maxlen=max_data_points)
setpoint_data = deque(maxlen=max_data_points)
position_data = deque(maxlen=max_data_points)
velocity_data = deque(maxlen=max_data_points)
loop_time_data = deque(maxlen=max_data_points)

plt.ion() # Turn on interactive mode.

# Create two subplots side-by-side, sharing the x-axis.
fig, (ax1, ax2, ax3)= plt.subplots(1, 3, figsize=(15,5), sharex=True)
fig.suptitle('ODrive Real-Time Control Data')

# Plot 1: Position and Setpoint.
line_setpoint, = ax1.plot([], [], 'r-', label='Setpoint (turns)')
line_position, = ax1.plot([], [], 'b-', label='Position (turns)')
ax1.legend()
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Position (turns)')
ax1.set_title('Position Control')
ax1.grid(True)

# Plot 2: Velocity
line_velocity, = ax2.plot([], [], 'g-', label='Velocity (turns/s)')
ax2.legend()
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Velocity (turns/s)')
ax2.set_title('Velocity')
ax2.grid(True)

# Plot 3: Loop Time
line_loop_time, = ax3.plot([], [], 'r-', label='Loop Time (s)')
ax3.legend()
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Loop Time (s)')
ax3.set_title('Loop Time')
ax3.grid(True)





####################################################
# --- MAIN SCRIPT ---
try:
    ####################################################
    # --- INITIALIZATION ---

    # Initialize CAN bus.
    bus = can.interface.Bus(channel=CHANNEL, interface=CAN_INTERFACE)
    print(f"Succesfully connected to {CHANNEL} using {CAN_INTERFACE}.")
    print(f"Configuring ODrive Axis {ODRIVE_AXIS_ID} for Position Control...")

    # Set Controller Modes: Position Control and Passthrough Input Mode.
    print(f"Sending: Set Axis {ODRIVE_AXIS_ID} to Control Mode {CONTROL_MODE_POSITION_CONTROL}, Input Mode {INPUT_MODE_POS_FILTER}")
    msg_set_modes = odrive_msg(ODRIVE_AXIS_ID, CMD_ID_SET_CONTROLLER_MODES,
                               (CONTROL_MODE_POSITION_CONTROL, INPUT_MODE_POS_FILTER), '<ii')
    if msg_set_modes: bus.send(msg_set_modes)
    time.sleep(0.5)

    # Set Axis State to Closed Loop Control.
    print(f"Sending: Set Axis {ODRIVE_AXIS_ID} to Closed Loop Control State ({AXIS_STATE_CLOSED_LOOP_CONTROL})")
    msg_set_state = odrive_msg(ODRIVE_AXIS_ID, CMD_ID_SET_AXIS_REQUESTED_STATE, AXIS_STATE_CLOSED_LOOP_CONTROL, '<i')
    if msg_set_state: bus.send(msg_set_state)
    time.sleep(0.2)

    print("--- WARNING: Assuming ODrive entered Closed Loop Control. ---")

    print(f"\nStarting Position Control.")
    print("Press Ctrl+C to stop or exit program.")

    start_time = time.time()
    loop_interval = 1.0 / CONTROL_LOOP_RATE_HZ

    ####################################################
    # --- MAIN LOOP ---
    while running:
        loop_start_time = time.time()
        current_time = time.time() - start_time

        ####################################################
        # --- CONTROL LOOP & DATA LOGGING ---
        if current_time < (SETPOINT_1_TIME + SETPOINT_2_TIME):

            # --- Recording Phase ---
            # Read all available messages in the buffer to get the latest one.
            latest_msg = None
            while True:
                msg = bus.recv(timeout=0) # Non-blocking read.
                if msg is None:
                    break # Buffer is empty.
                latest_msg = msg # Overwrite until we have the last message.

            # Use last known values as a default.
            current_position_turns = last_position_turns
            current_velocity_turns_per_sec = last_velocity_turns_per_sec

            # Process the latest message of we received one.
            if latest_msg is not None:
                expected_arb_id = (ODRIVE_AXIS_ID << 5) | CMD_ID_GET_ENCODER_ESTIMATES
                if latest_msg.arbitration_id == expected_arb_id and latest_msg.dlc == 8:
                    try:
                        pos_estimate, vel_estimate = struct.unpack('<ff', latest_msg.data)
                        current_position_turns = pos_estimate
                        current_velocity_turns_per_sec = vel_estimate
                        # Store the latest valid readings.
                        last_position_turns = current_position_turns
                        last_velocity_turns_per_sec = current_velocity_turns_per_sec
                    except struct.error as e:
                        print(f"Error unpacking encoder estimates: {e}", file=sys.stderr)

            # --- Setpoint Initialization ---
            if initialized == False:
                initial_setpoint = current_position_turns
                current_setpoint = initial_setpoint
                initialized = True

            # --- Step Change ---
            if current_time > SETPOINT_1_TIME:
                current_setpoint = initial_setpoint + STEP_SIZE

            # --- Command Phase ---
            msg_set_position = odrive_msg(ODRIVE_AXIS_ID, CMD_ID_SET_INPUT_POS, current_setpoint, '<f')
            if msg_set_position:
                bus.send(msg_set_position)

            # --- Record Data ---
            loop_time = time.time() - loop_start_time

            time_data.append(current_time)
            setpoint_data.append(current_setpoint)
            position_data.append(current_position_turns)
            velocity_data.append(current_velocity_turns_per_sec)
            loop_time_data.append(loop_time)

        ####################################################
        # --- PLOTTING ---
        else:
            # Plot graph and keep running until an exception is raised.
            plt.show()

            # Update plot data.
            line_setpoint.set_data(time_data, setpoint_data)
            line_position.set_data(time_data, position_data)
            line_velocity.set_data(time_data, velocity_data)
            line_loop_time.set_data(time_data, loop_time_data)

            # Adjust plot limits.
            ax1.set_xlim(0, (SETPOINT_1_TIME + SETPOINT_2_TIME))

            pos_plot_data = list(setpoint_data) + list(position_data)
            if pos_plot_data:
                min_y = min(pos_plot_data)
                max_y = max(pos_plot_data)
                data_range = max_y - min_y
                # If range is zero (flat line), create a default sensible range.
                if data_range == 0:
                    data_range = abs(max_y) if max_y != 0 else 1.0
                margin = data_range * 0.10
                ax1.set_ylim(min_y - margin, max_y + margin)

            # --- MODIFIED: Auto-scale y-axis for plot 2 (Velocity) with 10% margin. ---
            if velocity_data:
                min_y = min(velocity_data)
                max_y = max(velocity_data)
                data_range = max_y - min_y
                # If range is zero (flat line), create a default sensible range.
                if data_range == 0:
                    data_range = abs(max_y) if max_y != 0 else 1.0
                margin = data_range * 0.10
                ax2.set_ylim(min_y - margin, max_y + margin)

            # --- MODIFIED: Auto-scale y-axis for plot 3 (Loop Time) with 10% margin. ---
            if loop_time_data:
                min_y = min(loop_time_data)
                max_y = max(loop_time_data)
                data_range = max_y - min_y
                # If range is zero, use a small default since loop times are tiny.
                if data_range == 0:
                    data_range = abs(max_y) if max_y > 1e-6 else 0.01
                margin = data_range * 0.10
                ax3.set_ylim(min_y - margin, max_y + margin)

            fig.canvas.draw()
            fig.canvas.flush_events()

        # --- LOOP MAINTENANCE ---
        ####################################################

        # Maintain loop rate.
        elapsed_time = time.time() - loop_start_time
        sleep_time = loop_interval - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

####################################################
# --- INTERRUPTS ---

# --- Keyboard Interrupt ---
except KeyboardInterrupt:
    print("\nKeyboard Interrupt detected.")
# --- CAN Error ---
except can.CanError as e:
    print(f"\nCAN Error occurred: {e}", file=sys.stderr)
    print("Check connection, termination, bitrate.", file=sys.stderr)
# --- Unexpected Error ---
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
# --- Program Shutdown Requested ---
finally:
    print("\nExiting control loop.")
    if bus is not None:
        print("Setting motor to idle state.")
        msg_set_state = odrive_msg(ODRIVE_AXIS_ID, CMD_ID_SET_AXIS_REQUESTED_STATE, AXIS_STATE_IDLE, '<i')
        if msg_set_state:
            try:
                for _ in range(3): # Send multiple times.
                    bus.send(msg_set_state)
                    time.sleep(0.2)
            except can.CanError as e:
                print(f"Error sending motor idle command: {e}", file=sys.stderr)
        print("Shutting down CAN bus.")
        bus.shutdown()  
    print("Script finished.")