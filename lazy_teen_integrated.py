# lazy_teen_robot_integrated.py
import sys
sys.path.append('/home/group12/')
# --- IMPORTS ---
import cv2
import pickle
import numpy as np
import time
import pyrealsense2 as rs
import cv2.aruco as aruco

from maestro import Controller  # Ensure this file is named controller.py or update import
from faceRecognition import RealSenseFaceDetector as detector # Import the class
# --- DUMMY CLASSES/FUNCTIONS (Replace with your actual imports if these cause issues) ---
# This is to allow the rest of the script to be structured and runnable for review.
try:
    from maestro import Controller  # Your Maestro control library
except ImportError:
    print("Warning: Maestro Controller not found. Using dummy controller.")


    class Controller:
        def __init__(self, port='/dev/ttyACM0'):
            self.port = port
            print(f"Dummy Maestro Controller initialized on port {port}")

        def setTarget(self, channel, target):
            print(f"Dummy Maestro: Set Channel {channel} to {target}")

        def getPosition(self, channel):
            print(f"Dummy Maestro: GetPosition for Channel {channel} (returning neutral)")
            return 6000  # Standard neutral

        def close(self):
            print("Dummy Maestro: Closed")

try:
    # Assuming your faceRecognition.py has RealSenseFaceDetector class
    from faceRecognition import RealSenseFaceDetector
except ImportError:
    print("Warning: RealSenseFaceDetector from faceRecognition.py not found. Using dummy.")


    class RealSenseFaceDetector:
        # Making wait_for_consistent_face a static method as per your main.py usage
        @staticmethod
        def wait_for_consistent_face(duration=1, pipeline_object=None, face_size_threshold=(100, 100), display=True):
            print(f"Dummy Face Detection: Simulating wait for consistent face for {duration}s.")
            if display and pipeline_object:  # Try to show something if a pipeline is notionally passed
                try:
                    for _ in range(int(duration * 10)):  # Show dummy frames
                        frames = pipeline_object.wait_for_frames(500)
                        if frames:
                            color_frame = frames.get_color_frame()
                            if color_frame:
                                frame = np.asanyarray(color_frame.get_data())
                                cv2.putText(frame, "Dummy Face Search...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2)
                                cv2.imshow("Dummy Face Detection", frame)
                                if cv2.waitKey(100) & 0xFF == ord('s'):  # Allow manual skip
                                    print("Dummy Face Detection: Skipped by user.")
                                    cv2.destroyWindow("Dummy Face Detection")
                                    return True
                        else:
                            break
                    cv2.destroyWindow("Dummy Face Detection")
                except Exception as e:
                    print(f"Dummy display error: {e}")
            else:
                time.sleep(duration)  # Simple delay if no display/pipeline

            # Simulate face found for testing flow
            print("Dummy Face Detection: Consistent face 'detected'.")
            return True  # Simulate face detected for flow

# --- CONFIGURATION ---
# General
WIDTH = 640
HEIGHT = 480
FPS = 30
# ORB Recognizer (from your item_recognizer.py)
ORB_FEATURES = 1000
DIST_THRESH = 45
RATIO_THRESH = 0.75
COUNT_THRESH = 15  # Min good matches for confident object recognition
TRAINED_OBJECTS_FILE = 'trainedObjects.pkl'  #
# ArUco (from your lazy_teen.py and project doc)
ARUCO_DICT_NAME = aruco.DICT_4X4_50  # (Implied by IDs 0-4)
MARKER_SIZE_METERS = 0.1905  # IMPORTANT: From your code. MUST BE ACCURATE for pose.
CALIBRATION_FILE = '/home/group12/realsense_calibration_data.npz'
MARKER_ID_CENTER = 0  #
# Maestro Servos (from your lazy_teen.py)
MAESTRO_PORT = '/dev/ttyACM0'  # Update if necessary
SERVOS = {
    "wheels_both": 0,  # For forward/backward movement
    "wheels_opposite": 1,  # For turning (differential speed)
    "waist": 2,
    "head_up_down": 3,
    "head_side_to_side": 4,
    "right_arm_shoulder": 5,  # Example: Shoulder joint
    "right_arm_elbow": 6,  # Example: Elbow joint for raising [cite: 23]
    "right_arm_actuator": 7,  # Example: Gripper or mechanism for dropping ring [cite: 29]
    # Add other servos if needed
}
NEUTRAL = 6000  # Standard neutral servo position (1500 us)
# Face Detection
FACE_DETECTION_DURATION = 2  # seconds for consistent face implies it's a quick start
MIN_FACE_SIZE = (100, 100)  # pixels

# --- GLOBAL VARIABLES (Lazy Initialization) ---
pipeline = None
orb_detector = None  # Renamed from 'orb' to be more descriptive
bf_matcher = None  # Renamed from 'matcher'
trained_objects_data = None
aruco_detector_instance = None  # Renamed from 'detector' to avoid clash
maestro_controller = None
camera_matrix = None  # Renamed from 'mtx'
distortion_coeffs = None  # Renamed from 'dist'
face_detector_instance = RealSenseFaceDetector(
    width=WIDTH, height=HEIGHT, fps=FPS, external_pipeline=pipeline)
print(f"MODULE LEVEL: Initial face_detector_instance. Current id: {id(face_detector_instance)}")



# --- HELPER: TEXT-TO-SPEECH (Placeholder) ---
def speak(text):
    """Placeholder for Text-to-Speech."""
    print(f"\n🤖 LAZY TEEN SAYS: {text}\n")
    # In a real implementation, integrate a TTS library:
    # import pyttsx3
    # try:
    #     engine = pyttsx3.init()
    #     engine.say(text)
    #     engine.runAndWait()
    # except Exception as e:
    #     print(f"TTS Error: {e}")


# --- INITIALIZATION FUNCTIONS ---

def init_realsense_camera():
    global pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    try:
        print(f"Configuring RealSense: {WIDTH}x{HEIGHT} @ {FPS} FPS")
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        # config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS) # If depth is needed later
        profile = pipeline.start(config)
        device = profile.get_device()
        print(f"RealSense camera started. Connected to: {device.get_info(rs.camera_info.name)}")
        # Allow sensor to auto-exposure, etc.
        time.sleep(1)  # Wait for frames to stabilize
        return True
    except RuntimeError as e:
        speak(f"Ugh, my eyes are messed up. RealSense error: {e}")
        return False

# In lazy_teen_robot_integrated.py
def init_face_detector():
    global face_detector_instance, WIDTH, HEIGHT, FPS, pipeline # ADD `pipeline` here
    print(f"INIT_FACE_DETECTOR: Entered. Current face_detector_instance id: {id(face_detector_instance)}")
    if face_detector_instance is None: # Only create if not already created
        try:
            print("Creating RealSenseFaceDetector instance...")

            # Ensure the global pipeline (for the main script) is available.
            # init_realsense_camera() should have been called before this.
            if not pipeline:
                print("CRITICAL ERROR: Main RealSense pipeline not initialized before creating Face Detector.")
                speak("My main eyes aren't working, so I can't even try to look for faces.")
                return False

            # Pass the global `pipeline` to the RealSenseFaceDetector constructor
            face_detector_instance = RealSenseFaceDetector(
                width=WIDTH,
                height=HEIGHT,
                fps=FPS,
                external_pipeline=pipeline  # <--- PASSES THE MAIN PIPELINE
            )
            print("RealSenseFaceDetector instance created and configured with main pipeline.")
        except Exception as e:
            speak(f"Failed to create RealSenseFaceDetector instance: {e}")
            return False
    return True

def init_object_recognizer():
    global orb_detector, bf_matcher, trained_objects_data
    orb_detector = cv2.ORB_create(nfeatures=ORB_FEATURES)
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # crossCheck=False for knnMatch

    try:
        with open(TRAINED_OBJECTS_FILE, 'rb') as f:
            trained_objects_data = pickle.load(f)
        if not trained_objects_data:
            speak(f"My so-called memory ({TRAINED_OBJECTS_FILE}) is empty. Did you even bother training me?")
            return False
        print(f"Loaded trained data for {len(trained_objects_data)} objects from '{TRAINED_OBJECTS_FILE}'.")
        # Ensure IDs are integers if they are used as dictionary keys later
        for item in trained_objects_data:
            if 'id' in item: item['id'] = int(item['id'])
        return True
    except FileNotFoundError:
        speak(f"I can't find '{TRAINED_OBJECTS_FILE}'. Guess I don't know what to clean. Not my problem.")
        return False
    except Exception as e:
        speak(f"Loading '{TRAINED_OBJECTS_FILE}' failed: {e}. My brain hurts.")
        return False


def init_aruco_detection_system():
    global aruco_detector_instance, camera_matrix, distortion_coeffs
    try:
        with np.load(CALIBRATION_FILE) as data:
            camera_matrix = data['mtx']
            distortion_coeffs = data['dist']
        print(f"Camera calibration data loaded from '{CALIBRATION_FILE}'.")
    except FileNotFoundError:
        speak(f"Camera calibration '{CALIBRATION_FILE}' is missing. Everything's gonna look weird, don't blame me.")
        return False

    aruco_dictionary = aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
    aruco_parameters = aruco.DetectorParameters()
    aruco_detector_instance = aruco.ArucoDetector(aruco_dictionary, aruco_parameters)
    print(
        f"ArUco detector initialized for dictionary: {ARUCO_DICT_NAME}, Marker Size: {MARKER_SIZE_METERS * 1000:.1f}mm.")
    return True


def init_maestro_servo_controller():
    global maestro_controller
    try:
        maestro_controller = Controller(MAESTRO_PORT)
        # Optionally set speed/acceleration defaults here if your library supports it
        print(f"Maestro servo controller initialized on port {MAESTRO_PORT}.")
        return True
    except Exception as e:
        speak(f"I can't connect to my own muscles (Maestro error): {e}. Great.")
        return False


# --- ROBOT MOVEMENT FUNCTIONS (Adapted from your lazy_teen.py) ---
def set_servo_target(channel, target):
    if maestro_controller:
        maestro_controller.setTarget(channel, target)
    else:
        print(f"Debug: Servo {channel} to {target} (Maestro not connected)")


def move_forward_timed(duration, speed_pulse=4500):  # 4500 from your code for forward
    speak("Ugh, fine, moving forward.")
    set_servo_target(SERVOS["wheels_both"], speed_pulse)
    set_servo_target(SERVOS["wheels_opposite"], NEUTRAL)  # Ensure turning wheels are neutral
    time.sleep(duration)
    stop_all_movement()


def move_backward_timed(duration, speed_pulse=7500):  # Assuming 7500 for backward (NEUTRAL is 6000)
    speak("Backing up, I guess.")
    set_servo_target(SERVOS["wheels_both"], speed_pulse)
    set_servo_target(SERVOS["wheels_opposite"], NEUTRAL)
    time.sleep(duration)
    stop_all_movement()


def turn_left_timed(duration, turn_pulse=7000):  # 7000 from your code for left turn
    speak("Turning left. Are we there yet?")
    set_servo_target(SERVOS["wheels_opposite"], turn_pulse)  # Controls differential speed for turning
    set_servo_target(SERVOS["wheels_both"], NEUTRAL)  # Main drive wheels neutral or slow forward for pivot
    time.sleep(duration)
    stop_all_movement()


def turn_right_timed(duration, turn_pulse=5000):  # 5000 from your code for right turn
    speak("Turning right. This is thrilling.")
    set_servo_target(SERVOS["wheels_opposite"], turn_pulse)
    set_servo_target(SERVOS["wheels_both"], NEUTRAL)
    time.sleep(duration)
    stop_all_movement()


def stop_all_movement():
    # print("Stopping all movement.") # Can be too chatty
    set_servo_target(SERVOS["wheels_both"], NEUTRAL)
    set_servo_target(SERVOS["wheels_opposite"], NEUTRAL)
    # Could also neutralize head/waist if they were actively moving
    # set_servo_target(SERVOS["head_side_to_side"], NEUTRAL)


def look_left_timed(duration=0.5, look_target=8000):  # 8000 for left head turn
    set_servo_target(SERVOS["head_side_to_side"], look_target)
    time.sleep(duration)


def look_right_timed(duration=0.5, look_target=4000):  # 4000 for right head turn
    set_servo_target(SERVOS["head_side_to_side"], look_target)
    time.sleep(duration)


def look_center_timed(duration=0.3):
    set_servo_target(SERVOS["head_side_to_side"], NEUTRAL)
    time.sleep(duration)


def perform_arm_raise_for_ritual():  # [cite: 23]
    speak("Arm's up. Don't expect a parade.")
    # Replace with actual servo commands for your robot's arm
    # Example: move elbow servo to a "raised" position
    # This will depend on your robot's specific kinematics
    set_servo_target(SERVOS["right_arm_elbow"], 5000)  # Example: 5000 might be "elbow up"
    time.sleep(1.5)  # Allow time for arm to move
    print(">>> USER: Place the 'ring' on the robot's arm now. Waiting for 5 seconds. <<<")
    time.sleep(5)  # Give user time to place the object


def perform_ring_drop():  # [cite: 29, 30]
    speak("Aight, dropping this thing. Hope it lands somewhere.")
    # Replace with actual servo commands to drop the object
    # Example: open a gripper, or lower the arm further
    set_servo_target(SERVOS["right_arm_actuator"], 7000)  # Example: 7000 might be "open gripper"
    time.sleep(1)
    set_servo_target(SERVOS["right_arm_elbow"], NEUTRAL)  # Lower elbow back
    time.sleep(1)
    set_servo_target(SERVOS["right_arm_actuator"], NEUTRAL)  # Close gripper/reset actuator
    # Snarky comment if you want one [cite: 30, 31]
    speak("There. It's probably not in the box. Oh well.")


def reset_robot_to_neutral_stance():
    speak("Resetting myself. So much effort.")
    stop_all_movement()
    look_center_timed(0.5)
    set_servo_target(SERVOS["right_arm_elbow"], NEUTRAL)
    set_servo_target(SERVOS["right_arm_actuator"], NEUTRAL)
    # Add other servos (waist, shoulder) if they need resetting
    print("Robot servos set to neutral.")


# --- COMPUTER VISION AND ROBOT LOGIC FUNCTIONS ---
def wait_for_human_face_trigger(display=True):
    global face_detector_instance  # Use the global instance

    if not face_detector_instance:
        speak("Face detector was not initialized. Major oops.")
        return False

    speak("Ugh. Is someone there? Show your face or whatever.")

    # Call the method ON THE INSTANCE, and ONLY pass 'duration'
    face_found = face_detector_instance.wait_for_consistent_face(
        duration=FACE_DETECTION_DURATION
    )

    if face_found:
        speak("Ugh. What now?")
        return True
    else:
        speak("No face? Fine, I'm going back to ignoring everything.")
        return False

def identify_object_in_view(timeout_sec=15, display=True):  # [cite: 21, 22]
    global pipeline, orb_detector, bf_matcher, trained_objects_data
    speak("Alright, show me the junk I'm supposed to clean. Make it quick.")

    window_name = "Object Recognition"
    if display: cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    start_time = time.time()
    best_obj_name, best_obj_id = None, None

    while time.time() - start_time < timeout_sec:
        frames = pipeline.wait_for_frames(timeout_ms=1000)
        if not frames: continue
        color_frame = frames.get_color_frame()
        if not color_frame: continue

        frame = np.asanyarray(color_frame.get_data())
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_scene, des_scene = orb_detector.detectAndCompute(gray_frame, None)

        if des_scene is None or len(des_scene) < 2:  # Need at least 2 for knnMatch k=2
            if display:
                cv2.putText(frame, "Detecting...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        max_good_matches = 0

        for obj_data in trained_objects_data:
            obj_name_candidate = obj_data['name']
            obj_id_candidate = obj_data['id']  # This ID is the ArUco marker ID
            current_object_total_good_matches = 0

            for des_train in obj_data['descriptors']:
                if des_train is None or len(des_train) < 2: continue
                # Ensure des_train is CV_8U, ORB descriptors are binary
                if des_train.dtype != np.uint8: des_train = np.uint8(des_train)
                if des_scene.dtype != np.uint8: des_scene = np.uint8(des_scene)

                matches = bf_matcher.knnMatch(des_train, des_scene, k=2)

                good_matches_count_this_set = 0
                for m, n in matches:
                    if m.distance < RATIO_THRESH * n.distance and m.distance < DIST_THRESH:  # Adjusted Lowe's + absolute dist
                        good_matches_count_this_set += 1
                current_object_total_good_matches += good_matches_count_this_set

            if current_object_total_good_matches > max_good_matches:
                max_good_matches = current_object_total_good_matches
                best_obj_name = obj_name_candidate
                best_obj_id = obj_id_candidate

        display_label = "Detecting..."
        label_color = (0, 165, 255)  # Orange

        if best_obj_name and max_good_matches >= COUNT_THRESH:
            display_label = f"FOUND: {best_obj_name} (ID:{best_obj_id}) Matches: {max_good_matches}"
            label_color = (0, 255, 0)  # Green
            if display:
                cv2.putText(frame, display_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
                cv2.imshow(window_name, frame)
                cv2.waitKey(1)  # Show briefly
            speak(f"Fine. That's the {best_obj_name}. Guess I will put it in box #{best_obj_id}.")
            if display: cv2.destroyWindow(window_name)
            return best_obj_name, best_obj_id
        elif best_obj_name:
            display_label = f"Maybe: {best_obj_name} ({max_good_matches})"

        if display:
            cv2.putText(frame, display_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
            cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    speak("Couldn't recognize anything clearly. Or maybe I just zoned out.")
    if display: cv2.destroyWindow(window_name)
    return None, None


def navigate_to_aruco_marker(target_id, display=True):  # [cite: 26, 27, 28]
    global pipeline, aruco_detector_instance, camera_matrix, distortion_coeffs
    speak(f"Ugh, now I have to find marker {target_id}. This is the worst.")

    window_name = "ArUco Navigation"
    if display: cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Navigation parameters (these need extensive tuning on the actual robot!)
    TARGET_DISTANCE_Z = 0.35  # meters (how close to get to the marker)
    ACCEPTABLE_X_OFFSET = 0.05  # meters (how centered to be)
    MOVE_INCREMENT_DURATION = 0.4  # seconds for a short forward move
    TURN_INCREMENT_DURATION = 0.3  # seconds for a short turn
    SEARCH_TURN_DURATION = 0.6  # seconds for a wider search turn
    MAX_NAVIGATION_ATTEMPTS = 15  # Give up after this many tries

    # Known clockwise order of markers [cite: 15]
    FIELD_MARKERS_CLOCKWISE = [1, 2, 3, 4]  # Assuming these are the box markers

    found_initial_orientation_marker = False
    last_seen_marker_id = -1
    attempts = 0

    look_center_timed()  # Start by looking forward

    while attempts < MAX_NAVIGATION_ATTEMPTS:
        attempts += 1
        frames = pipeline.wait_for_frames(timeout_ms=1000)
        if not frames: continue
        color_frame = frames.get_color_frame()
        if not color_frame: continue

        frame = np.asanyarray(color_frame.get_data())
        corners, ids, _ = aruco_detector_instance.detectMarkers(frame)

        target_seen_this_frame = False

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_METERS, camera_matrix,
                                                              distortion_coeffs)

            for i, current_id_arr in enumerate(ids):
                current_id = current_id_arr[0]
                tvec = tvecs[i][0]  # (x, y, z) from camera to marker center
                rvec = rvecs[i][0]

                if display:
                    cv2.drawFrameAxes(frame, camera_matrix, distortion_coeffs, rvec, tvec, MARKER_SIZE_METERS * 0.5)
                    pos_str = f"ID:{current_id} X:{tvec[0]:.2f} Z:{tvec[2]:.2f}"
                    cv2.putText(frame, pos_str, tuple(corners[i][0][0].astype(int) - [0, 10]), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

                if not found_initial_orientation_marker and current_id in FIELD_MARKERS_CLOCKWISE:
                    found_initial_orientation_marker = True
                    last_seen_marker_id = current_id
                    speak(f"Spotted marker {current_id}. At least I know where *something* is.")

                if current_id == target_id:
                    target_seen_this_frame = True
                    dist_z = tvec[2]
                    offset_x = tvec[0]
                    speak(f"Target marker {target_id} sighted. Z:{dist_z:.2f}m, X:{offset_x:.2f}m.")

                    if dist_z < TARGET_DISTANCE_Z and abs(offset_x) < ACCEPTABLE_X_OFFSET:
                        stop_all_movement()
                        speak(f"Okay, I'm here at marker {target_id}. Close enough, right?")
                        if display: cv2.destroyWindow(window_name)
                        return True  # Successfully navigated

                    # --- Navigation Adjustments ---
                    if dist_z > TARGET_DISTANCE_Z + 0.05:  # Too far, move forward (main priority)
                        if abs(offset_x) > ACCEPTABLE_X_OFFSET * 2:  # Significantly off-center, turn first
                            if offset_x > ACCEPTABLE_X_OFFSET:
                                turn_left_timed(0.1)  # Marker is to our right
                            else:
                                turn_right_timed(0.1)  # Marker is to our left
                        move_forward_timed(MOVE_INCREMENT_DURATION * 0.7)  # Shorter burst if also turning
                    elif abs(offset_x) > ACCEPTABLE_X_OFFSET:  # Correct X offset if Z is okay
                        if offset_x > ACCEPTABLE_X_OFFSET:
                            turn_left_timed(TURN_INCREMENT_DURATION * 0.5)  # Marker is to our right, turn robot left
                        else:
                            turn_right_timed(TURN_INCREMENT_DURATION * 0.5)  # Marker is to our left, turn robot right
                    else:  # Minor adjustment if very close or slightly overshot Z
                        if dist_z < TARGET_DISTANCE_Z - 0.1:
                            move_backward_timed(0.2)  # Slightly too close
                        else:
                            move_forward_timed(0.1)  # Creep forward if slightly far but centered

                    break  # Processed target marker, new loop iteration for next frame
            # End of for loop for detected markers

        if not target_seen_this_frame:  # Target NOT seen in this frame
            speak("Can't see the target marker. Guess I'll look around.")
            if found_initial_orientation_marker and target_id in FIELD_MARKERS_CLOCKWISE and last_seen_marker_id in FIELD_MARKERS_CLOCKWISE:
                # Try to turn intelligently based on known layout [cite: 27]
                try:
                    current_idx = FIELD_MARKERS_CLOCKWISE.index(last_seen_marker_id)
                    target_idx = FIELD_MARKERS_CLOCKWISE.index(target_id)
                    diff = (target_idx - current_idx + len(FIELD_MARKERS_CLOCKWISE)) % len(FIELD_MARKERS_CLOCKWISE)

                    if diff == 1:  # Target is one step clockwise
                        speak(f"Thinking target {target_id} is to my right from {last_seen_marker_id}.")
                        turn_right_timed(SEARCH_TURN_DURATION * 0.7)
                    elif diff == len(FIELD_MARKERS_CLOCKWISE) - 1:  # Target is one step anti-clockwise
                        speak(f"Thinking target {target_id} is to my left from {last_seen_marker_id}.")
                        turn_left_timed(SEARCH_TURN_DURATION * 0.7)
                    else:  # Target is further away, wider turn
                        speak(f"Target {target_id} is probably a ways off from {last_seen_marker_id}. Bigger turn.")
                        turn_right_timed(SEARCH_TURN_DURATION)
                except ValueError:  # One of the IDs wasn't in the list (e.g. marker 0)
                    turn_right_timed(SEARCH_TURN_DURATION)  # Default search turn
            else:  # No orientation yet, or target is special (like marker 0)
                # Simple search pattern: look left, right, then turn body
                if attempts % 4 == 1:
                    look_left_timed(0.7)
                elif attempts % 4 == 2:
                    look_right_timed(1.4); look_center_timed(0.2)  # Full sweep then center
                else:
                    turn_right_timed(SEARCH_TURN_DURATION)

        if display:
            cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            speak("You quit navigation. Fine by me.")
            break
        time.sleep(0.3)  # Pause between navigation steps/attempts

    speak(f"Giving up on finding marker {target_id}. This is too much work.")
    if display: cv2.destroyWindow(window_name)
    return False


# --- MAIN APPLICATION SCRIPT ---
def run_robot_room_cleaner_demo():
    global pipeline  # Ensure it's accessible for cleanup

    # --- Initialization Phase ---
    # This part should run first to ensure the robot is ready
    speak("Ugh, guess I have to wake up now...")
    if not init_realsense_camera() or \
            not init_object_recognizer() or \
            not init_aruco_detection_system() or \
            not init_maestro_servo_controller() or \
            not init_face_detector():  # Ensure face detector is also initialized here
        speak("Something important didn't start. I'm going back to 'sleep'. Problem solved.")
        # Clean up any partial initializations
        if pipeline: pipeline.stop()
        if maestro_controller: maestro_controller.close()
        cv2.destroyAllWindows()
        return

    reset_robot_to_neutral_stance()  # Start from a known configuration

    # --- Phase 2: Room Cleaner Demo Protocol ---
    # 1. Wait for a Human Face
    if wait_for_human_face_trigger(display=True):  # If face IS detected
        # The "Ugh. What now?" is already spoken by wait_for_human_face_trigger on success

        # 2. Ask for the Object & 3. Identify the Object
        object_name, target_aruco_id_for_drop = identify_object_in_view(timeout_sec=20, display=True)

        if object_name and target_aruco_id_for_drop is not None:
            # 4. Wait for the Ring Ritual (Raise arm)
            perform_arm_raise_for_ritual()

            # 5. Find Correct ArUco Marker & 6. Move to Marker/Box
            if navigate_to_aruco_marker(target_aruco_id_for_drop, display=True):
                # (Verbal announcement about arrival is handled within navigate_to_aruco_marker if successful)
                # 6. Drop the Ring
                perform_ring_drop()
            else:
                speak(
                    f"Couldn't make it to marker {target_aruco_id_for_drop}. So, this thing stays with me. Your problem.")
        else:  # Object not identified or user quit
            speak("Didn't get an object to clean. So, I'm, like, done here.")

        # 7. Return to Start (Marker 0) - This now only happens if a face was detected and chore sequence was attempted
        speak("Alright, time to go back to doing nothing at the starting spot.")
        if navigate_to_aruco_marker(MARKER_ID_CENTER, display=True):
            speak("Made it back to the center. Nap time.")
        else:
            speak("Eh, couldn't find the exact center. This is good enough.")

        speak("Cleaning complete. Barely.")  # This message now makes more sense here

    else:  # Face was NOT detected by wait_for_human_face_trigger
        # The "No one around to boss me?..." message is already spoken by wait_for_human_face_trigger on failure
        # No chore sequence, no specific "Return to Start" for the chore is needed.
        # The robot will just proceed to the final reset and cleanup.
        pass  # Message already handled by wait_for_human_face_trigger

    # This final reset and cleanup should happen regardless of face detection success,
    # to ensure the robot is left in a safe state and resources are freed.
    reset_robot_to_neutral_stance()  # Ensure it's neutral at the very end.

    # --- Cleanup ---
    speak("Shutting down. Finally some peace.")  # General shutdown message
    if pipeline:
        print("Stopping RealSense pipeline...")
        pipeline.stop()
    if maestro_controller:
        print("Closing Maestro controller connection...")
        maestro_controller.close()
    cv2.destroyAllWindows()
    print("All windows closed. Robot script finished.")


if __name__ == "__main__":
    print("== LAZY TEENAGER ROBOT CLEANER - CSCI 442 FINAL PROJECT ==")
    print("Ensure:")
    print("  1. RealSense camera is connected.")
    print(f"  2. Trained objects file ('{TRAINED_OBJECTS_FILE}') exists.")
    print(f"  3. Camera calibration file ('{CALIBRATION_FILE}') exists.")
    print(f"  4. Maestro servo controller is connected to '{MAESTRO_PORT}'.")
    print("  5. All required Python libraries (OpenCV, PyRealSense, NumPy, etc.) are installed.")
    print("  6. Your `maestro.py` and `faceRecognition.py` are in the same directory or Python path.")
    print("\nPress 'q' in OpenCV windows to attempt to skip/quit certain stages if implemented.")
    print("----------------------------------------------------------")

    run_robot_room_cleaner_demo()