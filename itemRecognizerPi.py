import cv2
import pickle
import numpy as np
import time
import pyrealsense2 as rs # Import RealSense library

# --- Configuration ---
# Match settings used during training for consistency
WIDTH = 640
HEIGHT = 480
FPS = 30
# Use the same ORB feature count as during training
ORB_FEATURES = 1000

# Sensitivity parameters for matching
DIST_THRESH = 45    # Hamming distance threshold for Lowe's ratio test (adjust as needed)
RATIO_THRESH = 0.75 # Lowe's ratio test threshold (0.7-0.8 is common)
COUNT_THRESH = 15   # Minimum good match count to confidently label (adjust based on testing)

# --- Load Trained Data ---
try:
    with open('trainedObjects.pkl', 'rb') as f:
        trained_objects = pickle.load(f)
    if not trained_objects:
        print("Error: trainedObjects.pkl is empty or invalid.")
        exit(1)
    print(f"Loaded data for {len(trained_objects)} objects.")
except FileNotFoundError:
    print("Error: trainedObjects.pkl not found. Run the training script first.")
    exit(1)
except Exception as e:
    print(f"Error loading trainedObjects.pkl: {e}")
    exit(1)

# --- Initialize ORB and Matcher ---
# Use the same feature count as during training
orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
# Use BFMatcher with Hamming distance for ORB descriptors
# crossCheck=False is needed for knnMatch
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# --- Initialize RealSense ---
pipeline = rs.pipeline()
config = rs.config()

try:
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense devices connected.")
        exit(1)

    print(f"Configuring RealSense stream: {WIDTH}x{HEIGHT} @ {FPS} FPS")
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    profile = pipeline.start(config)
    print("RealSense stream started.")
    device = profile.get_device()
    print(f"Connected to: {device.get_info(rs.camera_info.name)}")

except RuntimeError as e:
    print(f"Error initializing RealSense: {e}")
    exit(1)
# --- RealSense Initialized ---

print("Starting recognition. Press 'q' to quit.")
recognition_window_name = "Recognition"
cv2.namedWindow(recognition_window_name)

try: # Ensure pipeline is stopped
    while True:
        frame = None
        start_time = time.time() # Optional: for performance timing

        # --- Get frame from RealSense ---
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000) # Timeout
            if not frames:
                time.sleep(0.05)
                continue
            color_frame = frames.get_color_frame()
            if not color_frame:
                time.sleep(0.05)
                continue
            frame = np.asanyarray(color_frame.get_data())
        except RuntimeError as e:
             print(f"RealSense Error during recognition loop: {e}")
             time.sleep(0.1)
             continue # Try to recover in the next iteration
        # --- Frame Acquired ---

        if frame is None: continue # Should not happen with checks, but safety first

        # --- Object Recognition Logic ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect ORB features in the current frame
        kp_frame, des_frame = orb.detectAndCompute(gray, None)

        best_match_name = None
        best_match_count = 0

        if des_frame is not None and len(des_frame) > 0:
            # Compare current frame descriptors against all trained objects
            for obj in trained_objects:
                current_object_good_matches = 0
                # Iterate through all descriptor sets captured for this object during training
                for trained_des in obj['descriptors']:
                    if trained_des is None or len(trained_des) == 0:
                         continue # Skip empty descriptors if any slipped through training

                    # Match frame descriptors against this specific set of trained descriptors
                    # k=2 for Lowe's ratio test
                    matches = matcher.knnMatch(trained_des, des_frame, k=2)

                    # Apply Lowe's ratio test to filter good matches
                    for m, n in matches:
                        # Check distances: m is the best match, n is the second best
                        # Good if distance is below threshold AND significantly better than second best
                        if m.distance < DIST_THRESH and m.distance < RATIO_THRESH * n.distance:
                           current_object_good_matches += 1

                # Keep track of the object with the most good matches overall
                if current_object_good_matches > best_match_count:
                    best_match_count = current_object_good_matches
                    best_match_name = obj['name']
        # --- End Recognition Logic ---


        # --- Display Results ---
        label = "Unknown"
        color = (0, 0, 255) # Red for unknown

        if best_match_name:
            if best_match_count >= COUNT_THRESH:
                label = f"{best_match_name} ({best_match_count})"
                color = (0, 255, 0) # Green for confident match
            else:
                # Display potential match below confidence threshold
                label = f"Maybe: {best_match_name} ({best_match_count})"
                color = (0, 165, 255) # Orange for low confidence
        # else: label remains "Unknown"

        # Optional: Display FPS for performance monitoring
        end_time = time.time()
        fps = 1.0 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        cv2.putText(frame, f"Detected: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow(recognition_window_name, frame)
        # --- End Display ---

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally: # Ensure resources are released
    print("Stopping RealSense pipeline...")
    pipeline.stop()
    print("RealSense stopped.")
    cv2.destroyAllWindows()
    print("Recognition finished.")