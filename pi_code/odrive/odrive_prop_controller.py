#!/usr/bin/env python3

import can
import struct
import time
import sys
import signal # For graceful exit

########################################################################
# CONTROLLER PARAMETERS

TARGET_SETPOINT = 0.0  # Target position in turns

# --- PID GAINS - TUNING REQUIRED ---
# Start with Ki = 0, Kd = 0 and tune Kp first.
# Then tune Kd for damping. Tune Ki last (if needed) for steady-state error.
K_P = 0.32 # Proportional Gain (Amps / Turn)
K_D = 0.00 # Derivative Gain (Amps / (Turn/Sec))
K_I = 0.00 # Integral Gain (Amps / (Turn*Sec))

# --- SAFETY LIMIT ---
MAX_CURRENT_AMPS = 2.0 # Max allowed current command

# --- LOOP TIMING ---
CONTROL_LOOP_RATE_HZ = 100 # Control loop frequency (e.g., 100Hz = 10ms interval)

########################################################################

# --- Configuration ---
CAN_INTERFACE = 'socketcan'
CHANNEL = 'can0'
ODRIVE_AXIS_ID = 4  # Match ODrive's configured Node ID for Axis 0

# ODrive CAN Command IDs
CMD_ID_SET_AXIS_REQUESTED_STATE = 0x007
CMD_ID_SET_CONTROLLER_MODES = 0x00B
CMD_ID_SET_INPUT_TORQUE = 0x00E # Used to command current in current control mode
CMD_ID_GET_ENCODER_ESTIMATES = 0x009 # Used to receive Pos/Vel estimates

# ODrive Axis States
AXIS_STATE_CLOSED_LOOP_CONTROL = 8

# ODrive Control Modes
CONTROL_MODE_CURRENT_CONTROL = 1

# ODrive Input Modes
INPUT_MODE_PASSTHROUGH = 1

# --- Globals for clean exit & PID state ---
bus = None
running = True
last_position_turns = 0.0
last_velocity_turns_per_sec = 0.0
error_integral = 0.0 # Integral accumulator

# --- Helper function to pack and create CAN message ---
# (Same as before)
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

# --- Main Script ---
try:
    # Initialize CAN bus
    bus = can.interface.Bus(channel=CHANNEL, interface=CAN_INTERFACE)
    print(f"Successfully connected to {CHANNEL} using {CAN_INTERFACE}.")
    print(f"Configuring ODrive Axis {ODRIVE_AXIS_ID} for Current Control...")

    # 1. Set Axis State to Closed Loop Control
    print(f"Sending: Set Axis {ODRIVE_AXIS_ID} to Closed Loop Control State ({AXIS_STATE_CLOSED_LOOP_CONTROL})")
    msg_set_state = odrive_msg(ODRIVE_AXIS_ID, CMD_ID_SET_AXIS_REQUESTED_STATE, AXIS_STATE_CLOSED_LOOP_CONTROL, '<i')
    if msg_set_state: bus.send(msg_set_state)
    time.sleep(0.2)

    # 2. Set Controller Modes: Current Control and Passthrough Input Mode
    print(f"Sending: Set Axis {ODRIVE_AXIS_ID} to Control Mode {CONTROL_MODE_CURRENT_CONTROL}, Input Mode {INPUT_MODE_PASSTHROUGH}")
    msg_set_modes = odrive_msg(ODRIVE_AXIS_ID, CMD_ID_SET_CONTROLLER_MODES,
                               (CONTROL_MODE_CURRENT_CONTROL, INPUT_MODE_PASSTHROUGH), '<ii')
    if msg_set_modes: bus.send(msg_set_modes)
    time.sleep(0.2)

    print("--- WARNING: Assuming ODrive entered Closed Loop Control. ---")

    print(f"\nStarting PID Position Control Loop (Target: {TARGET_SETPOINT:.3f} turns, Kp: {K_P:.3f}, Ki: {K_I:.3f}, Kd: {K_D:.3f})")
    print("Press Ctrl+C to stop.")

    last_print_time = time.time()
    loop_interval = 1.0 / CONTROL_LOOP_RATE_HZ
    error_integral = 0.0 # Reset integral just before loop starts

    # --- Control Loop ---
    while running:
        loop_start_time = time.time()

        # --- Feedback Phase ---
        # (Removed GET_ENCODER_ESTIMATES send - assuming ODrive sends periodically)
        # Attempt to receive the latest feedback message


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


        # --- Control Calculation Phase ---
        error = TARGET_SETPOINT - current_position_turns

        # Proportional Term
        p_term = K_P * error

        # Derivative Term (acts on velocity)
        d_term = - (K_D * current_velocity_turns_per_sec) # Negative velocity = positive error derivative

        # Integral Term (accumulates error) - Calculated using value from *previous* step
        i_term = K_I * error_integral

        # Combine terms to get desired current before clamping
        target_current_unclamped = p_term + i_term + d_term

        # Clamp the output current to safe limits
        target_current_clamped = max(-MAX_CURRENT_AMPS, min(MAX_CURRENT_AMPS, target_current_unclamped))

        # --- Anti-Windup for Integral Term ---
        # Only accumulate integral error if the output is NOT saturated at the limits
        if abs(target_current_clamped) < MAX_CURRENT_AMPS:
             error_integral += error * loop_interval
        # Optional: Clamp the integral accumulator itself to prevent unbounded growth
        # integral_limit = 5.0 # Example limit (Amps / Ki)
        # error_integral = max(-integral_limit, min(integral_limit, error_integral))


        # --- Command Phase ---
        # Send the CLAMPED current command to ODrive
        msg_set_current = odrive_msg(ODRIVE_AXIS_ID, CMD_ID_SET_INPUT_TORQUE, target_current_clamped, '<f')
        if msg_set_current:
            bus.send(msg_set_current)

        # --- Loop Maintenance ---
        # Print status periodically
        current_time = time.time()
        if current_time - last_print_time >= 0.1: # Print roughly 10 times/sec
             print(f"\rTgt: {TARGET_SETPOINT:.2f} Pos: {current_position_turns:.3f} Err: {error:.3f} Int: {error_integral:.3f} Vel: {current_velocity_turns_per_sec:.2f} CmdI: {target_current_clamped:.2f}A", end="")
             last_print_time = current_time

        # Maintain loop rate
        elapsed_time = time.time() - loop_start_time
        sleep_time = loop_interval - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nKeyboard Interrupt detected.")
except can.CanError as e:
    print(f"\nCAN Error occurred: {e}", file=sys.stderr)
    print("Check connection, termination, bitrate.", file=sys.stderr)
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
finally:
    print("\nExiting control loop.")
    if bus is not None:
        # Send 0 Amp command before shutting down for safety
        print("Sending 0 Amp current command.")
        msg_zero_current = odrive_msg(ODRIVE_AXIS_ID, CMD_ID_SET_INPUT_TORQUE, 0.0, '<f')
        if msg_zero_current:
            try:
                for _ in range(3): # Send multiple times
                    bus.send(msg_zero_current)
                    time.sleep(0.01)
            except can.CanError as e:
                 print(f"Error sending zero current command: {e}", file=sys.stderr)

        print("Shutting down CAN bus.")
        bus.shutdown()
    print("Script finished.")
