# lazy_teen_robot_integrated.py
import sys

sys.path.append('/home/group12/')  # Ensure this is correct
# --- IMPORTS ---
import cv2
import pickle
import numpy as np
import time
import pyrealsense2 as rs
import cv2.aruco as aruco
import threading  # <<< --- ADD THIS LINE HERE ---

# Assuming face_detector_module.py is in the Python path (e.g. /home/group12/)
# or in the same directory
from faceRecognition import RealSenseFaceDetector  # Corrected import if file is named face_detector_module.py

# --- DUMMY CLASSES/FUNCTIONS (Keep your existing dummies or real imports) ---
try:
    from maestro import Controller
except ImportError:
    print("Warning: Maestro Controller not found. Using dummy controller.")


    class Controller:
        def __init__(self, port='/dev/ttyACM0'): self.port = port; print(
            f"Dummy Maestro Controller initialized on port {port}")

        def setTarget(self, channel, target): print(f"Dummy Maestro: Set Channel {channel} to {target}")

        def getPosition(self, channel): print(
            f"Dummy Maestro: GetPosition for Channel {channel} (returning neutral)"); return 6000

        def close(self): print("Dummy Maestro: Closed")

# --- CONFIGURATION (Keep your existing configurations) ---
WIDTH = 640
HEIGHT = 480
FPS = 30
ORB_FEATURES = 1000
DIST_THRESH = 45
RATIO_THRESH = 0.75
COUNT_THRESH = 15
TRAINED_OBJECTS_FILE = 'trainedObjects.pkl'
ARUCO_DICT_NAME = aruco.DICT_4X4_50
MARKER_SIZE_METERS = 0.1905
CALIBRATION_FILE = '/home/group12/realsense_calibration_data.npz'
MARKER_ID_CENTER = 0
MAESTRO_PORT = '/dev/ttyACM0'
SERVOS = {"wheels_both": 0, "wheels_opposite": 1, "waist": 2, "head_up_down": 3, "head_side_to_side": 4,
          "right_arm_shoulder": 5, "right_arm_elbow": 6, "right_arm_actuator": 7}
NEUTRAL = 6000
OBJECT_CONSISTENCY_DURATION = 1.0  # NEW: Seconds an object must be consistently detected
FACE_DETECTION_DURATION = 1
MIN_FACE_SIZE = (100, 100)

# --- GLOBAL VARIABLES ---
pipeline = None
orb_detector = None
bf_matcher = None
trained_objects_data = None
aruco_detector_instance = None
maestro_controller = None
camera_matrix = None
distortion_coeffs = None
face_detector_instance = None  # Will be initialized in init_face_detector
movement_thread = None  # To hold the thread object
_movement_thread_active_flag = threading.Event()


# --- HELPER: TEXT-TO-SPEECH ---
def speak(text):
    print(f"\n🤖 LAZY TEEN SAYS: {text}\n")


# --- INITIALIZATION FUNCTIONS (largely unchanged, but init_face_detector passes pipeline) ---

def init_realsense_camera():
    global pipeline
    print("INIT_REALSENSE_CAMERA: Entered.")
    if pipeline is not None:  # If a pipeline object already exists
        try:
            if pipeline.get_active_profile():  # Check if it's active
                print(
                    "INIT_REALSENSE_CAMERA: Pipeline already exists and is active. Stopping it before re-initializing.")
                pipeline.stop()
            else:
                print("INIT_REALSENSE_CAMERA: Pipeline object exists but is not active.")
        except RuntimeError as e:
            print(f"INIT_REALSENSE_CAMERA: Error checking/stopping existing pipeline: {e}. Proceeding to create new.")
        except Exception as e:  # Catch any other error during pre-check
            print(f"INIT_REALSENSE_CAMERA: Unexpected error with existing pipeline object: {e}. Setting to None.")
        pipeline = None  # Ensure it's None before creating a new one

    try:
        print("INIT_REALSENSE_CAMERA: Attempting to create rs.pipeline().")
        pipeline = rs.pipeline()
        print(
            f"INIT_REALSENSE_CAMERA: rs.pipeline() created. Global pipeline is now type: {type(pipeline)}, id: {id(pipeline)}")
        config = rs.config()
        print(f"INIT_REALSENSE_CAMERA: Configuring stream {WIDTH}x{HEIGHT} @ {FPS} FPS.")
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        print("INIT_REALSENSE_CAMERA: Attempting pipeline.start(config).")
        profile = pipeline.start(config)
        device = profile.get_device()
        print(
            f"INIT_REALSENSE_CAMERA: SUCCESS - RealSense camera started. Connected to: {device.get_info(rs.camera_info.name)}.")
        print(
            f"INIT_REALSENSE_CAMERA: Global pipeline id: {id(pipeline)} is active: {pipeline.get_active_profile() is not None}.")
        time.sleep(0.5)  # Shorter sleep after start
        return True
    except RuntimeError as e:
        speak(f"Ugh, my eyes are messed up. RealSense error in init_realsense_camera: {e}")
        print(
            f"INIT_REALSENSE_CAMERA: RuntimeError. Global pipeline (at error): type {type(pipeline)}, id {id(pipeline)}.")
        if pipeline:  # If pipeline object exists but start failed
            try:
                pipeline.stop()  # Try to stop it to free resources
            except:
                pass
        pipeline = None  # Ensure it's None on failure
        return False
    except Exception as e:
        speak(f"Unexpected error in init_realsense_camera: {e}")
        print(
            f"INIT_REALSENSE_CAMERA: Unexpected error. Pipeline (before error) id: {id(pipeline if 'pipeline' in locals() or 'pipeline' in globals() else None)}.")
        pipeline = None
        return False


def init_face_detector():
    global face_detector_instance, WIDTH, HEIGHT, FPS, pipeline
    print("INIT_FACE_DETECTOR: Entered.")
    if pipeline is None:
        speak("INIT_FACE_DETECTOR: My main eyes (RealSense pipeline) aren't working. Can't initialize face detector.")
        print("INIT_FACE_DETECTOR: RealSense pipeline is None. Aborting face detector initialization.")
        return False

    print(f"INIT_FACE_DETECTOR: Global 'pipeline' received is type: {type(pipeline)}, id: {id(pipeline)}")
    try:
        # Always create a new instance, passing the *current* global pipeline
        # The RealSenseFaceDetector is designed to USE this external pipeline
        print("INIT_FACE_DETECTOR: Attempting to create new RealSenseFaceDetector instance...")
        face_detector_instance = RealSenseFaceDetector(
            width=WIDTH,
            height=HEIGHT,
            fps=FPS,
            external_pipeline=pipeline  # Pass the shared pipeline
        )
        print(f"INIT_FACE_DETECTOR: Instance CREATED. New face_detector_instance id: {id(face_detector_instance)}")
        # The RealSenseFaceDetector's _initialize_realsense will check if the passed pipeline is active.
        # The main script is responsible for pipeline.start()
        return True
    except Exception as e:
        speak(f"INIT_FACE_DETECTOR: FAILED to create RealSenseFaceDetector instance: {e}")
        print(f"INIT_FACE_DETECTOR: Error during creation. face_detector_instance set to None.")
        face_detector_instance = None
        return False


def init_object_recognizer():
    global orb_detector, bf_matcher, trained_objects_data
    orb_detector = cv2.ORB_create(nfeatures=ORB_FEATURES)
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        with open(TRAINED_OBJECTS_FILE, 'rb') as f:
            trained_objects_data = pickle.load(f)
        if not trained_objects_data: speak(f"My memory ({TRAINED_OBJECTS_FILE}) is empty."); return False
        print(f"Loaded trained data for {len(trained_objects_data)} objects.")
        for item in trained_objects_data:
            if 'id' in item: item['id'] = int(item['id'])
        return True
    except FileNotFoundError:
        speak(f"Can't find '{TRAINED_OBJECTS_FILE}'."); return False
    except Exception as e:
        speak(f"Loading '{TRAINED_OBJECTS_FILE}' failed: {e}."); return False


def init_aruco_detection_system():
    global aruco_detector_instance, camera_matrix, distortion_coeffs
    try:
        with np.load(CALIBRATION_FILE) as data:
            camera_matrix, distortion_coeffs = data['mtx'], data['dist']
        print(f"Camera calibration data loaded from '{CALIBRATION_FILE}'.")
    except FileNotFoundError:
        speak(f"Calibration '{CALIBRATION_FILE}' missing."); return False
    aruco_dictionary = aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
    aruco_parameters = aruco.DetectorParameters()
    aruco_detector_instance = aruco.ArucoDetector(aruco_dictionary, aruco_parameters)
    print(f"ArUco detector initialized.")
    return True


def init_maestro_servo_controller():
    global maestro_controller
    try:
        maestro_controller = Controller(MAESTRO_PORT)
        print(f"Maestro servo controller initialized on port {MAESTRO_PORT}.")
        return True
    except Exception as e:
        speak(f"Maestro error: {e}."); return False


# --- ROBOT MOVEMENT FUNCTIONS (Keep your existing movement functions) ---
def set_servo_target(channel, target):
    if maestro_controller: maestro_controller.setTarget(channel, target)
    # else: print(f"Debug: Servo {channel} to {target} (Maestro not connected)")


def move_forward_timed(duration, speed_pulse=4500):
    speak("Moving forward.");
    set_servo_target(SERVOS["wheels_both"],
    speed_pulse); set_servo_target(
    SERVOS["wheels_opposite"], NEUTRAL); time.sleep(duration); stop_all_movement()


def move_backward_timed(duration, speed_pulse=7500):
    speak("Backing up.");
    set_servo_target(SERVOS["wheels_both"],
    speed_pulse); set_servo_target(
    SERVOS["wheels_opposite"], NEUTRAL); time.sleep(duration); stop_all_movement()


def turn_left_timed(duration, turn_pulse=7000):
    speak("Turning left.")
    set_servo_target(SERVOS["wheels_opposite"],
    turn_pulse); set_servo_target(
    SERVOS["wheels_both"], NEUTRAL)
    time.sleep(duration)
    stop_all_movement()


def turn_right_timed(duration, turn_pulse=5000):
    speak("Turning right.");
    set_servo_target(SERVOS["wheels_opposite"],
    turn_pulse); set_servo_target(
    SERVOS["wheels_both"], NEUTRAL); time.sleep(duration); stop_all_movement()


def stop_all_movement():
    set_servo_target(SERVOS["wheels_both"], NEUTRAL)
    set_servo_target(SERVOS["wheels_opposite"],NEUTRAL)


def look_left_timed(duration=0.5, look_target=8000):
    set_servo_target(SERVOS["head_side_to_side"],look_target)
    time.sleep(duration)


def look_right_timed(duration=0.5, look_target=4000):
    set_servo_target(SERVOS["head_side_to_side"],look_target)
    time.sleep(duration)


def look_center_timed(duration=0.3):
    set_servo_target(SERVOS["head_side_to_side"], NEUTRAL)
    time.sleep(duration)


def perform_arm_raise_for_ritual():
    speak("Arm's up.")
    set_servo_target(SERVOS["right_arm_shoulder"], 5000)
    time.sleep(1.5); print(">>> USER: Place 'ring'. Waiting 5s. <<<"); time.sleep(5)

def perform_ring_drop():
    speak("Dropping this.")
    set_servo_target(SERVOS["right_arm_actuator"], 7000)
    time.sleep(1)
    set_servo_target(SERVOS["right_arm_shoulder"], NEUTRAL); time.sleep(1)
    set_servo_target(SERVOS["right_arm_actuator"], NEUTRAL); speak("There. Not in the box. Oh well.")


def reset_robot_to_neutral_stance(): speak("Resetting."); stop_all_movement(); look_center_timed(0.5); set_servo_target(
    SERVOS["right_arm_shoulder"], NEUTRAL); set_servo_target(SERVOS["right_arm_actuator"], NEUTRAL); print(
    "Robot servos neutral.")


# --- PIPELINE SHUTDOWN ---
def shutdown_realsense_camera():
    global pipeline
    print("SHUTDOWN_REALSENSE_CAMERA: Entered.")
    if pipeline:
        try:
            if pipeline.get_active_profile():  # Important check
                print(f"SHUTDOWN_REALSENSE_CAMERA: Pipeline active. Stopping pipeline id {id(pipeline)}.")
                pipeline.stop()
                print("SHUTDOWN_REALSENSE_CAMERA: Pipeline stopped.")
            else:
                print("SHUTDOWN_REALSENSE_CAMERA: Pipeline was not active, no need to stop.")
        except RuntimeError as e:
            print(f"SHUTDOWN_REALSENSE_CAMERA: RuntimeError stopping pipeline: {e}")
        except Exception as e:
            print(f"SHUTDOWN_REALSENSE_CAMERA: Unexpected error stopping pipeline: {e}")
        finally:
            pipeline = None; print("SHUTDOWN_REALSENSE_CAMERA: Global pipeline variable set to None.")
    else:
        print("SHUTDOWN_REALSENSE_CAMERA: Global pipeline was already None.")
    return True


# --- COMPUTER VISION AND ROBOT LOGIC FUNCTIONS ---

def wait_for_human_face_trigger(display=True):
    global face_detector_instance  # Uses the global instance initialized by init_face_detector
    if not face_detector_instance:
        speak("Face detector was not initialized. Major oops.")
        return False

    speak("Ugh. Is someone there? Show your face or whatever.")
    # Call wait_for_consistent_face on the INSTANCE.
    # This is now a BLOCKING call that will create and manage its own OpenCV window.
    # It uses the pipeline passed to it during its construction.
    face_found = face_detector_instance.wait_for_consistent_face(
        duration=FACE_DETECTION_DURATION,
        display_window_name="LazyTeen: Face Check"  # Optional: give it a unique name
    )

    if face_found:
        speak("Ugh. What now?")
        return True
    else:
        speak("No face? Fine, I'm going back to ignoring everything.")
        return False


def identify_object_in_view(timeout_sec=15, display=True):
    global pipeline, orb_detector, bf_matcher, trained_objects_data, OBJECT_CONSISTENCY_DURATION, COUNT_THRESH
    speak("Alright, show me the junk. Make it quick. And hold it steady for like, a second.")

    if not pipeline or not pipeline.get_active_profile():
        speak("My eyes aren't working (pipeline not active). Can't see objects.")
        return None, None

    window_name = "LazyTeen: Object Recognition"
    if display:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        except Exception as e:
            print(f"DEBUG: FAILED TO CREATE WINDOW '{window_name}': {e}"); display = False

    start_time = time.time()

    # Variables for consistency check
    consistent_detection_start_time = None
    current_consistent_object_name = None
    current_consistent_object_id = None

    while time.time() - start_time < timeout_sec:
        frames = None
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError as e:
            print(f"DEBUG: RealSense RuntimeError in object_in_view: {e}. Skipping frame.")
            time.sleep(0.1)
            if "Frame didn't arrive within" not in str(e): break
            continue

        if not frames: continue
        color_frame = frames.get_color_frame()
        if not color_frame: continue

        frame = np.asanyarray(color_frame.get_data())
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_scene, des_scene = orb_detector.detectAndCompute(gray_frame, None)

        # Best object found in *this current frame*
        best_name_this_frame = None
        best_id_this_frame = None
        max_matches_this_frame = 0

        if des_scene is not None and len(des_scene) >= 2:
            for obj_data in trained_objects_data:
                obj_name_candidate = obj_data.get('name', 'Unknown')
                obj_id_candidate = obj_data.get('id', -1)
                trained_descriptors_list = obj_data.get('descriptors')
                if trained_descriptors_list is None: continue

                current_object_total_good_matches = 0
                for des_train in trained_descriptors_list:
                    if des_train is None or len(des_train) < 2: continue
                    if des_train.dtype != np.uint8: des_train = np.uint8(des_train)

                    des_scene_uint8 = des_scene
                    if des_scene.dtype != np.uint8: des_scene_uint8 = np.uint8(des_scene)

                    try:
                        matches = bf_matcher.knnMatch(des_train, des_scene_uint8, k=2)
                    except cv2.error:
                        continue

                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < RATIO_THRESH * n.distance and m.distance < DIST_THRESH:
                                current_object_total_good_matches += 1

                if current_object_total_good_matches > max_matches_this_frame:
                    max_matches_this_frame = current_object_total_good_matches
                    best_name_this_frame = obj_name_candidate
                    best_id_this_frame = obj_id_candidate

        # --- Consistency Logic ---
        display_label = "Detecting..."
        label_color = (0, 165, 255)  # Orange for detecting

        if best_name_this_frame and max_matches_this_frame >= COUNT_THRESH:
            # An object meeting criteria is seen in this frame
            if current_consistent_object_name == best_name_this_frame:
                # Same object as before, check duration
                elapsed_consistent_time = time.time() - consistent_detection_start_time
                if elapsed_consistent_time >= OBJECT_CONSISTENCY_DURATION:
                    # CONFIRMED!
                    display_label = f"CONFIRMED: {best_name_this_frame} (ID:{best_id_this_frame})"
                    label_color = (0, 255, 0)  # Green for confirmed
                    if display:
                        cv2.putText(frame, display_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
                        try:
                            cv2.imshow(window_name, frame)
                        except Exception as e:
                            print(f"DEBUG: imshow failed (confirmed): {e}"); display = False
                        cv2.waitKey(1)  # Show final confirmation
                    speak(f"Fine. That's the {best_name_this_frame}. Guess I will put it in box #{best_id_this_frame}.")
                    if display:
                        try:
                            cv2.destroyWindow(window_name); cv2.waitKey(1)
                        except Exception as e:
                            print(f"DEBUG: destroyWindow '{window_name}' failed: {e}")
                    return best_name_this_frame, best_id_this_frame
                else:
                    # Still tracking, duration not met
                    display_label = f"Tracking: {best_name_this_frame} ({elapsed_consistent_time:.1f}s / {OBJECT_CONSISTENCY_DURATION:.1f}s)"
                    label_color = (0, 255, 255)  # Yellow for tracking
            else:
                # New potential object, or different from last tracked one
                print(f"DEBUG: New potential object for consistency: {best_name_this_frame}")
                consistent_detection_start_time = time.time()
                current_consistent_object_name = best_name_this_frame
                current_consistent_object_id = best_id_this_frame
                display_label = f"Tracking: {best_name_this_frame} (0.0s / {OBJECT_CONSISTENCY_DURATION:.1f}s)"
                label_color = (0, 255, 255)  # Yellow for tracking
        else:
            # No object met COUNT_THRESH in this frame, or no descriptors
            if current_consistent_object_name:
                print(f"DEBUG: Lost consistent track of {current_consistent_object_name}. Resetting.")
            consistent_detection_start_time = None
            current_consistent_object_name = None
            current_consistent_object_id = None
            if best_name_this_frame:  # Show if something was weakly detected
                display_label = f"Maybe: {best_name_this_frame} ({max_matches_this_frame} matches)"
                label_color = (0, 165, 255)  # Orange

        if display:
            cv2.putText(frame, display_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
            try:
                cv2.imshow(window_name, frame)
            except Exception as e:
                print(f"DEBUG: imshow failed (loop end): {e}"); display = False

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): print("DEBUG: 'q' pressed. Exiting object ID."); break
        if display:
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break
            except:
                break

    speak("Couldn't recognize anything consistently. Or maybe I just zoned out.")
    if display:
        try:
            cv2.destroyWindow(window_name); cv2.waitKey(1)
        except Exception as e:
            print(f"DEBUG: destroyWindow '{window_name}' after timeout failed: {e}")
    return None, None

def _execute_timed_movement_in_thread(movement_function_to_call, movement_duration, *args_for_movement_function):
    """
    Wrapper to execute a timed movement function (e.g., turn_left_timed) in a thread.
    Manages the _movement_thread_active_flag.
    """
    global _movement_thread_active_flag
    try:
        # Optional: print(f"THREAD: Starting {movement_function_to_call.__name__} for {movement_duration}s")
        movement_function_to_call(movement_duration, *args_for_movement_function) # This function handles its own speak, servo commands, sleep, and stop
        # Optional: print(f"THREAD: Finished {movement_function_to_call.__name__}")
    except Exception as e:
        print(f"THREAD: Error during movement {movement_function_to_call.__name__}: {e}")
    finally:
        _movement_thread_active_flag.clear() # Signal that movement is complete

def start_non_blocking_timed_movement(movement_function_ref, duration_sec, *args_for_movement):
    global movement_thread, _movement_thread_active_flag

    if _movement_thread_active_flag.is_set():
        # print("DEBUG: A non-blocking movement is already active. Skipping new request.")
        return False # Indicate movement was not started because one is already in progress

    _movement_thread_active_flag.set() # Signal that a movement is now starting/active

    # Clean up any previous thread object if it's finished
    if movement_thread and not movement_thread.is_alive():
        movement_thread.join() # Ensure resources are released

    movement_thread = threading.Thread(
        target=_execute_timed_movement_in_thread,
        args=(movement_function_ref, duration_sec, *args_for_movement)
    )
    movement_thread.daemon = True # Allows the main program to exit even if this thread is running
    movement_thread.start()
    # print(f"MAIN: Started non-blocking movement: {movement_function_ref.__name__} for {duration_sec}s")
    return True # Indicate movement was successfully started

def navigate_to_aruco_marker(target_id, display=True):
    global pipeline, aruco_detector_instance, camera_matrix, distortion_coeffs
    global _movement_thread_active_flag, movement_thread  # For managing threaded turns

    speak(f"Ugh, find marker {target_id}.")

    if not pipeline or not pipeline.get_active_profile():
        speak("My eyes aren't working (pipeline not active). Can't navigate.")
        print("DEBUG: navigate_to_aruco_marker - pipeline not active or None.")
        return False

    window_name = "LazyTeen: ArUco Navigation"
    if display:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        except Exception as e:
            print(f"DEBUG: Failed to create window '{window_name}': {e}");
            display = False

    TARGET_DISTANCE_Z = 0.35
    ACCEPTABLE_X_OFFSET = 0.05
    MOVE_INCREMENT_DURATION = 0.3
    TURN_INCREMENT_DURATION = 0.2
    SEARCH_TURN_DURATION = 0.5  # Default duration for search turns
    MAX_NAVIGATION_ATTEMPTS = 20

    FIELD_MARKERS_CLOCKWISE = [1, 2, 3, 4]
    found_initial_orientation_marker = False
    last_seen_marker_id = -1
    attempts = 0

    # --- Threading & Cooldown Variables ---
    last_non_blocking_turn_command_time = 0.0  # Time of the last non-blocking turn command
    # Cooldown: use the longest typical search turn duration + a buffer
    # The `attempts % 5 == 1` case uses 1.2s, so base cooldown on that.
    LONGEST_SEARCH_TURN_DURATION = 1.2
    NON_BLOCKING_TURN_COOLDOWN = LONGEST_SEARCH_TURN_DURATION + 0.4  # Cooldown before another non-blocking turn can start

    look_center_timed()  # Initial head centering, blocking is fine here.

    while attempts < MAX_NAVIGATION_ATTEMPTS:
        attempts += 1
        current_frame_time = time.time()  # For cooldown logic
        frames = None
        try:
            # Consider reducing timeout_ms for faster loop if CPU allows and frames are arriving quickly
            frames = pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError as e:
            print(f"DEBUG: RealSense RuntimeError in ArUco nav: {e}. Skipping frame.")
            time.sleep(0.05)  # Short sleep on error
            # More robust error check:
            if "Frame didn't arrive within" not in str(e) and "No Frames" not in str(e) and "Timed out" not in str(
                    e).lower():
                break  # Exit on more serious or persistent errors
            continue

        if not frames: continue
        color_frame = frames.get_color_frame()
        if not color_frame: continue

        frame = np.asanyarray(color_frame.get_data())
        corners, ids, _ = aruco_detector_instance.detectMarkers(frame)
        target_seen_this_frame = False

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            try:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_METERS, camera_matrix,
                                                                  distortion_coeffs)
            except cv2.error as e:
                print(f"DEBUG: ArUco pose estimation error: {e}. Skipping pose this frame.")
                rvecs, tvecs = None, None

            if rvecs is not None and tvecs is not None:
                for i, current_id_arr in enumerate(ids):
                    current_id = current_id_arr[0]
                    tvec = tvecs[i][0]
                    rvec = rvecs[i][0]

                    if display:  # Drawing logic (unchanged)
                        try:
                            cv2.drawFrameAxes(frame, camera_matrix, distortion_coeffs, rvec, tvec,
                                              MARKER_SIZE_METERS * 0.5)
                            pos_str = f"ID:{current_id} X:{tvec[0]:.2f} Z:{tvec[2]:.2f}"
                            if corners[i] is not None and len(corners[i][0]) > 0:
                                text_origin = tuple(corners[i][0][0].astype(int) - np.array([0, 10]))
                                cv2.putText(frame, pos_str, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        except Exception as e:
                            print(f"DEBUG: Error drawing ArUco details: {e}")

                    if not found_initial_orientation_marker and current_id in FIELD_MARKERS_CLOCKWISE:
                        found_initial_orientation_marker = True
                        last_seen_marker_id = current_id
                        speak(f"Spotted marker {current_id}.")

                    if current_id == target_id:
                        target_seen_this_frame = True
                        dist_z = tvec[2]
                        offset_x = tvec[0]

                        # --- Target Seen: Stop non-blocking turn and use precise blocking moves ---
                        if _movement_thread_active_flag.is_set():
                            speak("Target sighted! Halting search turn for precise alignment.")
                            stop_all_movement()
                            # The non-blocking thread will eventually clear its flag.
                            # Subsequent movements are blocking and precise.

                        if dist_z < TARGET_DISTANCE_Z and abs(offset_x) < ACCEPTABLE_X_OFFSET:
                            stop_all_movement()
                            speak(f"Okay, I'm here at marker {target_id}.")
                            if display:
                                try:
                                    cv2.destroyWindow(window_name); cv2.waitKey(1)
                                except Exception as e:
                                    print(f"DEBUG: destroyWindow '{window_name}' failed: {e}")
                            return True  # SUCCESS

                        # Precise (blocking) movements:
                        if dist_z > TARGET_DISTANCE_Z + 0.05:
                            if abs(offset_x) > ACCEPTABLE_X_OFFSET * 1.5:
                                if offset_x > 0:
                                    turn_left_timed(TURN_INCREMENT_DURATION * 0.6)
                                else:
                                    turn_right_timed(TURN_INCREMENT_DURATION * 0.6)
                            else:
                                move_forward_timed(MOVE_INCREMENT_DURATION * 0.7)
                        elif abs(offset_x) > ACCEPTABLE_X_OFFSET:
                            if offset_x > 0:
                                turn_left_timed(TURN_INCREMENT_DURATION * 0.5)
                            else:
                                turn_right_timed(TURN_INCREMENT_DURATION * 0.5)
                        elif dist_z < TARGET_DISTANCE_Z - 0.1:
                            move_backward_timed(0.2)
                        else:
                            move_forward_timed(0.1)  # Creep

                        break  # Processed target, break from 'for ids' loop to get new frame after movement
        # End of 'if ids is not None:'

        # --- Target NOT Seen: Consider starting a non-blocking search turn ---
        if not target_seen_this_frame:
            turn_initiated_this_cycle = False
            if not _movement_thread_active_flag.is_set() and \
                    (current_frame_time - last_non_blocking_turn_command_time > NON_BLOCKING_TURN_COOLDOWN):

                # speak("Target not seen. Considering a non-blocking turn.") # Optional debug

                # 1. Try smart turning based on field markers
                if found_initial_orientation_marker and target_id in FIELD_MARKERS_CLOCKWISE and last_seen_marker_id in FIELD_MARKERS_CLOCKWISE:
                    try:
                        current_idx = FIELD_MARKERS_CLOCKWISE.index(last_seen_marker_id)
                        target_idx = FIELD_MARKERS_CLOCKWISE.index(target_id)
                        diff = (target_idx - current_idx + len(FIELD_MARKERS_CLOCKWISE)) % len(FIELD_MARKERS_CLOCKWISE)

                        if diff == 1:  # Target likely to robot's right
                            speak(f"Target {target_id} might be right. Non-blocking right turn.")
                            if start_non_blocking_timed_movement(turn_right_timed, SEARCH_TURN_DURATION * 0.6):
                                turn_initiated_this_cycle = True
                        elif diff == len(FIELD_MARKERS_CLOCKWISE) - 1:  # Target likely to robot's left
                            speak(f"Target {target_id} might be left. Non-blocking left turn.")
                            if start_non_blocking_timed_movement(turn_left_timed, SEARCH_TURN_DURATION * 0.6):
                                turn_initiated_this_cycle = True
                        else:  # Target further away
                            speak(f"Target {target_id} far. Wider non-blocking right turn.")
                            if start_non_blocking_timed_movement(turn_right_timed,
                                                                 SEARCH_TURN_DURATION):  # Default wider turn
                                turn_initiated_this_cycle = True
                    except ValueError:  # Should not happen if logic is sound
                        speak("Error in field marker logic. Defaulting to generic non-blocking turn.")
                        if start_non_blocking_timed_movement(turn_right_timed, SEARCH_TURN_DURATION):
                            turn_initiated_this_cycle = True
                else:
                    # 2. Generic search pattern (replaces old head looks and some body turns)
                    if attempts % 5 == 1:
                        speak("Generic search: Non-blocking long right turn.")
                        if start_non_blocking_timed_movement(turn_right_timed, 1.2):  # Your specified 1.2s duration
                            turn_initiated_this_cycle = True
                    elif attempts % 5 == 0:
                        speak("Generic search: Non-blocking standard right body turn.")
                        if start_non_blocking_timed_movement(turn_right_timed, SEARCH_TURN_DURATION):
                            turn_initiated_this_cycle = True
                    # For other attempts % 5 values (2, 3, 4), no turn is initiated here,
                    # allowing the robot to just scan while stationary for a few cycles.

                if turn_initiated_this_cycle:
                    last_non_blocking_turn_command_time = current_frame_time  # Update timestamp

            # else if _movement_thread_active_flag.is_set():
            # print("DEBUG: Non-blocking turn in progress. Vision loop continues.") # Optional
            # else:
            # print(f"DEBUG: Cooldown active. Wait {NON_BLOCKING_TURN_COOLDOWN - (current_frame_time - last_non_blocking_turn_command_time):.2f}s for next turn.") # Optional

        if display:
            try:
                cv2.imshow(window_name, frame)
            except Exception as e:
                print(f"DEBUG: imshow failed in ArUco nav: {e}");
                display = False

        key = cv2.waitKey(1) & 0xFF  # Essential for OpenCV window and keeping loop responsive
        if key == ord('q'): speak("Navigation quit by user."); break
        if display:
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break  # Window closed
            except:  # Error checking window, assume closed
                break

        # Removed the general time.sleep(0.1) from here.
        # Loop speed is now governed by frame acquisition, processing, waitKey(1), and turn cooldowns.

    # --- End of while loop ---
    speak(f"Giving up on marker {target_id} after {attempts} attempts.")

    # --- Cleanup before exiting function ---
    if _movement_thread_active_flag.is_set():
        speak("Navigation function ending. Stopping any active search movement.")
        stop_all_movement()  # Ensure motors are stopped
        if movement_thread and movement_thread.is_alive():
            # print("DEBUG: Waiting for active movement thread to complete its cycle...")
            movement_thread.join(timeout=LONGEST_SEARCH_TURN_DURATION + 0.2)  # Wait for thread to finish
        _movement_thread_active_flag.clear()  # Explicitly clear flag

    if display:
        try:
            cv2.destroyWindow(window_name);
            cv2.waitKey(1)
        except Exception as e:
            print(f"DEBUG: destroyWindow '{window_name}' after timeout failed: {e}")
    return False

# --- MAIN APPLICATION SCRIPT ---
def run_robot_room_cleaner_demo():
    global pipeline, face_detector_instance, maestro_controller

    speak("Ugh, guess I have to wake up now...")
    # Initialize camera FIRST, as it's needed by face detector
    if not init_realsense_camera():  # This starts the pipeline
        speak("Main eyes (RealSense) failed to start. I'm done.")
        shutdown_realsense_camera()  # Attempt cleanup
        cv2.destroyAllWindows()
        return

    # Now initialize other components that might need the pipeline or other resources
    # Note: init_face_detector() uses the global 'pipeline'
    if not init_object_recognizer() or \
            not init_aruco_detection_system() or \
            not init_maestro_servo_controller() or \
            not init_face_detector():  # init_face_detector now relies on an active pipeline
        speak("Something important didn't start. Going back to 'sleep'.")
        shutdown_realsense_camera()
        if maestro_controller:
            try:
                maestro_controller.close()
            except:
                pass
        cv2.destroyAllWindows()
        return

    reset_robot_to_neutral_stance()

    # wait_for_human_face_trigger will now use the *already active* global `pipeline`
    # and manage its own OpenCV window sequentially.
    if wait_for_human_face_trigger(display=True):
        print("DEBUG: Main script - Returned from wait_for_human_face_trigger.")
        # The face trigger function now handles its own window creation/destruction.
        # cv2.destroyAllWindows() here might be redundant if face_trigger cleaned up,
        # but it's a good safety measure before the next vision task.
        cv2.destroyAllWindows()
        cv2.waitKey(50)  # Short delay to ensure window context switches if needed

        # identify_object_in_view will create its own window.
        object_name, target_aruco_id_for_drop = identify_object_in_view(timeout_sec=25, display=True)
        cv2.destroyAllWindows()  # Clean up after object identification
        cv2.waitKey(50)

        if object_name and target_aruco_id_for_drop is not None:
            perform_arm_raise_for_ritual()
            # navigate_to_aruco_marker will create its own window.
            if navigate_to_aruco_marker(target_aruco_id_for_drop, display=True):
                perform_ring_drop()
            else:
                speak(f"Couldn't make it to marker {target_aruco_id_for_drop}.")
            cv2.destroyAllWindows()  # Clean up after navigation
            cv2.waitKey(50)
        else:
            speak("Didn't get an object to clean. Done.")

        speak("Alright, going back to start.")
        # navigate_to_aruco_marker will create its own window.
        if navigate_to_aruco_marker(MARKER_ID_CENTER, display=True):
            speak("Made it back to center. Nap time.")
        else:
            speak("Eh, couldn't find center. Good enough.")
        cv2.destroyAllWindows()  # Final vision task cleanup
        cv2.waitKey(50)
        speak("Cleaning complete. Barely.")
    else:
        speak("No human, no work. Perfect.")

    reset_robot_to_neutral_stance()
    speak("Shutting down. Finally peace.")
    shutdown_realsense_camera()  # This stops the pipeline
    if maestro_controller:
        print("Closing Maestro controller connection...")
        maestro_controller.close()
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Final waitKey for OpenCV
    print("All windows closed. Robot script finished.")


if __name__ == "__main__":
    print("Ensure:")
    print(f"  1. RealSense camera connected. Target port: {MAESTRO_PORT}")
    print(f"  2. Trained objects file ('{TRAINED_OBJECTS_FILE}') exists.")
    print(f"  3. Camera calibration ('{CALIBRATION_FILE}') exists.")
    print("  4. Maestro controller connected.")
    print("  5. Libraries (OpenCV, PyRealSense, NumPy, etc.) installed.")
    print("  6. `face_detector_module.py` (or your equivalent) is accessible.")
    print("\nPress 'q' in OpenCV windows to attempt to skip/quit stages.")
    print("----------------------------------------------------------")
    try:
        run_robot_room_cleaner_demo()
    except Exception as e:
        print(f"FATAL ERROR in main execution: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Ensure resources are released even if an unhandled exception occurs
        print("Executing final cleanup from __main__ finally block...")
        if 'pipeline' in globals() and pipeline:
            try:
                if pipeline.get_active_profile(): pipeline.stop()
            except:
                print("Error stopping pipeline in main finally.")
            pipeline = None
        if 'maestro_controller' in globals() and maestro_controller:
            try:
                maestro_controller.close()
            except:
                print("Error closing maestro in main finally.")
        cv2.destroyAllWindows()
        print("Main finally block cleanup complete.")