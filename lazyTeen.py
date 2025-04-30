import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import time
from maestro import Controller  # Ensure this file is named controller.py or update import
from faceRecognition import RealSenseFaceDetector as detector # Import the class


# --- 1. Load Calibration Data ---
calibration_file = 'realsense_calibration_data.npz'
try:
    with np.load(calibration_file) as data:
        mtx = data['mtx']
        dist = data['dist']
    print(f"Calibration data loaded successfully from {calibration_file}.")
except FileNotFoundError:
    print(f"Error: Calibration data file '{calibration_file}' not found.")
    print("Please run the RealSense calibration script first.")
    exit()

# --- 2. Configure RealSense Stream ---
pipeline = rs.pipeline()
config = rs.config()

# Try to enable 640x480 color stream. Adjust if needed.
width, height, fps = 640, 480, 30
try:
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    print(f"Enabled color stream: {width}x{height} @ {fps}fps")
except RuntimeError as e:
    print(f"Error enabling specified stream: {e}. Check supported resolutions.")
    exit()

# Start streaming
try:
    pipeline.start(config)
    print("RealSense pipeline started.")
except RuntimeError as e:
    print(f"Error starting RealSense pipeline: {e}")
    exit()

# --- 3. ArUco Setup ---
# !!! IMPORTANT: Change this dictionary to match your actual printed markers !!!
# Examples: aruco.DICT_6X6_250, aruco.DICT_4X4_50, aruco.DICT_5X5_100 etc.
# The project doc mentioned IDs 0-15 [cite: 12] - check which dictionary they belong to.
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# !!! IMPORTANT: Measure your printed markers' side length (in meters) !!!
# Example: A 55mm marker is 0.055 meters. Update this value!
marker_size_meters = 0.1905  # Adjust to your measured marker size in meters!

print("Starting ArUco detection loop... Press 'q' to quit.")
print(f"Looking for markers from dictionary: {aruco_dict}")
print(f"Assuming marker size: {marker_size_meters * 1000:.1f} mm")

# Initialize the Maestro controller
maestro = Controller('/dev/ttyACM0')  # Update port if necessary

# Define neutral and test positions
NEUTRAL = 6000  # Standard neutral position (1.5ms pulse)
SMALL_MOVE = 5500  # Small test movement
REVERSE_MOVE = 6500  # Opposite small movement

# Servo assignments based on your list
SERVOS = {
    "wheels_both": 0,
    "wheels_opposite": 1,
    "waist": 2,
    "head_up_down": 3,
    "head_side_to_side": 4,
    "right_arm": list(range(5, 10)),  # Shoulder to hand
    "left_arm": list(range(11, 16)),  # Shoulder to hand
}


def move_forward(runtime):
    print("Moving forwards")
    maestro.setTarget(SERVOS["wheels_both"], 4500)
    time.sleep(runtime)
    stop_robot()


def stop_robot():
    print("stopping robot")
    maestro.setTarget(SERVOS["wheels_both"], NEUTRAL)
    maestro.setTarget(SERVOS["wheels_opposite"], NEUTRAL)


def stop_head():
    print("stopping head")
    maestro.setTarget(SERVOS["head_side_to_side"], NEUTRAL)


def reset_robot():
    print("Resetting robot.")  # Optional print
    maestro.setTarget(SERVOS["wheels_both"], NEUTRAL)
    maestro.setTarget(SERVOS["wheels_opposite"], NEUTRAL)
    maestro.setTarget(SERVOS["head_side_to_side"], NEUTRAL)
    print("robot reset")


def turn_left(runtime):
    print("  Executing turn_left() maneuver...")
    # Your sequence using maestro.setTarget for channel 0 and 1 7050 ideal for left turn
    maestro.setTarget(SERVOS["wheels_opposite"], 7000)
    print("  turn_left() finished.")
    time.sleep(runtime)
    stop_robot()


def turn_right(runtime):
    print("Executing turn_right() maneuver...")
    # Your sequence using maestro.setTarget for channel 0 and 1 4950 ideal for right turn
    maestro.setTarget(SERVOS["wheels_opposite"], 5000);
    print("  turn_right() finished.")
    time.sleep(runtime)
    stop_robot()


def look_left():
    print("Execute left head turn")
    maestro.setTarget(SERVOS["head_side_to_side"], 8000)
    print("head_turn_left complete")
    time.sleep(0.5)


def look_right():
    print("Execute right head turn")
    maestro.setTarget(SERVOS["head_side_to_side"], 4000)
    print("Head_turn_right complete")
    time.sleep(0.5)

def spin_search(runtime):
    print("Executing turn_right() maneuver...")
    # Your sequence using maestro.setTarget for channel 0 and 1 4950 ideal for right turn
    maestro.setTarget(SERVOS["wheels_opposite"], 5000);
    print("  turn_right() finished.")
    time.sleep(runtime)
    stop_robot()

def get_face():
    required_consistency = 1  # seconds

    print(f"\nAttempting face detection pre-check (need {required_consistency}s consistency)...")

    # *** THIS IS THE CORRECT CALL ***
    face_detected = detector.wait_for_consistent_face(duration=required_consistency)

    # --- Proceed Based on Detection Result ---
    if face_detected:
        print("\nSUCCESS: Consistent face detected!")
        print("Proceeding with the next series of calls...")
        # ADD YOUR OTHER FUNCTION CALLS HERE
    else:
        print("\nFAILURE: Face detection did not meet consistency requirement or was stopped manually.")
        print("Cannot proceed with subsequent calls.")

try:  # Use try...finally to ensure pipeline is stopped
    while True:
        # --- 4. Get Image Frame ---
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("Warning: No color frame received. Skipping frame.")
            time.sleep(0.1)
            continue

        # Convert to OpenCV format
        frame = np.asanyarray(color_frame.get_data())
        if frame is None or frame.size == 0:
            print("Warning: Received empty frame. Skipping.")
            continue

        # --- 5. Detect Markers ---
        corners, ids, rejectedImgPoints = detector.detectMarkers(frame)

        # --- 6. Process Detections ---
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Estimate pose for each detected marker [cite: 3]
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size_meters, mtx, dist)

            for i in range(len(ids)):
                id_num = ids[i][0]
                rvec = rvecs[i]
                tvec = tvecs[i]  # Translation vector (x, y, z) in meters from camera

                # Draw axis for visualization (length is half the marker size)
                cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, marker_size_meters * 0.5)

                # Display ID and Position (X, Y as required by project doc [cite: 7])
                pos_str = f"ID: {id_num} X: {tvec[0][0]:.2f}m Y: {tvec[0][1]:.2f}m Z: {tvec[0][2]:.2f}m"
                print(pos_str)  # Print to console as well

                # Put text on the image near the marker
                corner_set = corners[i]
                corner_top_left = tuple(corner_set[0][0].astype(int))
                cv2.putText(frame, pos_str, (corner_top_left[0], corner_top_left[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # --- Placeholder for Navigation/Control Logic ---
                # Here you would add your code to:
                # 1. Check if ID is odd/even [cite: 5]
                # 2. Determine left/right pass [cite: 5]
                # 3. Send commands to robot motors based on tvec (position)
                # 4. Send commands to pan/tilt servos based on marker position in frame [cite: 4]
                # 5. Update robot coordinates [cite: 21]
                # 6. Check for stopping condi# --- Navigation Control (Inside the 'if ids is not None:' block, inside the 'for i in range(len(ids)):' loop) ---

                # -------------------------------------------------
                # --- Navigation Control (Inside the 'if ids is not None:' block, inside the 'for i in range(len(ids)):' loop) ---

                # 1. Determine Pass Side & Target X Offset
                # TARGET_OFFSET_X needs tuning! Determines how far robot passes to the side.
                if id_num % 2 == 0:  # Even ID
                    print("even number")

                else:  # Odd ID
                    print("odd number")


                # -------------------------------------------------


        else:
            # Optional: Display message if no markers detected
            cv2.putText(frame, "No markers detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('RealSense ArUco Detection', frame)

        # Exit loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:  # Ensure cleanup
    print("Stopping RealSense pipeline...")
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Pipeline stopped and windows closed.")
