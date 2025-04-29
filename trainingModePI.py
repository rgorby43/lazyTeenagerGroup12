import cv2
import pickle
import numpy as np
import time
import pyrealsense2 as rs  # Import RealSense library

# --- Configuration ---
# Adjust resolution/FPS based on Pi 4 performance and camera model
# Lower resolutions (e.g., 640x480) are generally better for Pi performance
WIDTH = 640
HEIGHT = 480
FPS = 30
# Lower ORB features might be needed for Pi 4 performance
ORB_FEATURES = 1000 # Start with 1000, reduce if lagging (e.g., 500)
# Delay in seconds during auto-capture (adjust based on Pi performance)
CAPTURE_DELAY = 0.05 # Smaller delay for potentially faster capture

# Globals for mouse callback
refPt = []
cropping = False
mx, my = -1, -1  # mouse position during drag


def click_and_crop(event, x, y, flags, param):
    """Mouse callback: record bounding box start/end and track dragging."""
    global refPt, cropping, mx, my
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        mx, my = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        x1, y1 = refPt[0]
        x2, y2 = x, y
        # Store as top-left and bottom-right points
        refPt = [(min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2))]
        cropping = False


def draw_box(img):
    """Draw current bounding box on image."""
    img_copy = img.copy() # Draw on a copy
    if len(refPt) == 1 and cropping:
        cv2.rectangle(img_copy, refPt[0], (mx, my), (0, 255, 0), 2)
    elif len(refPt) == 2:
        cv2.rectangle(img_copy, refPt[0], refPt[1], (0, 0, 255), 2) # Final selection box
    return img_copy


def auto_capture_tracked(pipeline, orb, num_samples=30, delay=0.1):
    """
    Automatically capture samples while tracking ROI via a CSRT tracker, using RealSense.
    Returns list of descriptor arrays.
    """
    global refPt # Use the globally set refPt from selection phase

    if len(refPt) != 2:
        print("Error: Bounding box not selected correctly before auto_capture.")
        return []

    # Get initial frame for tracker initialization
    initial_frame = None
    print("Waiting for initial frame for tracking...")
    for _ in range(30): # Try for a few seconds to get a frame
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000) # Wait up to 1 sec
            if frames:
                color_frame = frames.get_color_frame()
                if color_frame:
                    initial_frame = np.asanyarray(color_frame.get_data())
                    break # Got the frame
            print("  ...")
            time.sleep(0.1)
        except RuntimeError as e:
            print(f"Error getting initial frame: {e}. Retrying...")
            time.sleep(0.5)
    if initial_frame is None:
        print("Failed to get initial frame for tracking.")
        return []


    (x1, y1), (x2, y2) = refPt
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        print("Error: Invalid bounding box dimensions.")
        return []

    # --- Create CSRT tracker ---
    tracker = None
    try:
        # CSRT is generally preferred for accuracy, but computationally heavier.
        # If performance on Pi4 is poor, consider cv2.TrackerKCF_create()
        tracker = cv2.TrackerCSRT_create()
    except AttributeError:
         # CSRT might be in legacy depending on OpenCV version
        try:
            tracker = cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            print("Error: TrackerCSRT not found. Check OpenCV (contrib) installation.")
            return []
    # --- Tracker Selected ---

    try:
        tracker.init(initial_frame, (x1, y1, w, h))
        print(f"Tracker initialized with box: x={x1}, y={y1}, w={w}, h={h}")
    except Exception as e:
        print(f"Error initializing tracker: {e}")
        return []

    descriptors = []
    captured_count = 0
    frame_count = 0
    max_attempts = num_samples * 10 # Allow more attempts for potential frame drops/tracking failures

    capture_window_name = "Training - Auto Capture"
    cv2.namedWindow(capture_window_name)

    while captured_count < num_samples and frame_count < max_attempts:
        frame_count += 1
        frame = None
        try:
            # --- Get frame from RealSense ---
            frames = pipeline.wait_for_frames(timeout_ms=500) # Shorter timeout in loop
            if not frames:
                # print("Warning: No frames received.") # Reduce console spam
                time.sleep(0.05)
                continue
            color_frame = frames.get_color_frame()
            if not color_frame:
                # print("Warning: No color frame.") # Reduce console spam
                time.sleep(0.05)
                continue
            frame = np.asanyarray(color_frame.get_data())
            # --- Frame Acquired ---

        except RuntimeError as e:
             print(f"RealSense Error during capture loop: {e}")
             time.sleep(0.1)
             continue

        if frame is None: continue

        ok, bbox = tracker.update(frame)
        disp = frame.copy()

        if ok:
            x, y, w, h = map(int, bbox)

            # Ensure ROI coordinates are valid
            if w > 0 and h > 0 and x >= 0 and y >= 0 and (x + w) <= frame.shape[1] and (y + h) <= frame.shape[0]:
                roi = frame[y:y+h, x:x+w]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                kp, des = orb.detectAndCompute(gray_roi, None)

                if des is not None and len(des) > 0: # Check if descriptors are found
                    descriptors.append(des)
                    captured_count += 1
                    cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 0), 2) # Green: success

                    # Visualize Keypoints (can be slow on Pi, disable if needed)
                    kp_disp = []
                    for p in kp:
                        kp_disp.append(cv2.KeyPoint(x=p.pt[0] + x, y=p.pt[1] + y, size=p.size,
                                                   angle=p.angle, response=p.response, octave=p.octave,
                                                   class_id=p.class_id))
                    cv2.drawKeypoints(disp, kp_disp, disp, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    #----------------------------------------------------------

                else:
                    cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 255), 2) # Yellow: no features
                    # cv2.putText(disp, "No features", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 0, 255), 2) # Red: invalid ROI dims
                # cv2.putText(disp, "Invalid ROI", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else: # Tracking failed
            if frame_count % 15 == 0: # Print failure message less often
                 print(f"Tracking failure {frame_count}, captured {captured_count}/{num_samples}")
            cv2.putText(disp, "Tracking Failure", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display progress
        cv2.putText(disp, f"{captured_count}/{num_samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(capture_window_name, disp)

        # Use waitKey for UI events AND delay. Crucial for Pi responsiveness.
        key_pressed = cv2.waitKey(max(1, int(delay * 1000))) & 0xFF
        if key_pressed == ord('q'):
            print("Capture interrupted by user.")
            break

    if captured_count < num_samples:
        print(f"Warning: Only captured {captured_count} out of {num_samples} desired samples.")

    cv2.destroyWindow(capture_window_name)
    return descriptors


def main():
    # --- Initialize RealSense ---
    pipeline = rs.pipeline()
    config = rs.config()

    try:
        # Check for connected devices
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("No RealSense devices connected.")
            return

        # Enable color stream
        print(f"Configuring RealSense stream: {WIDTH}x{HEIGHT} @ {FPS} FPS")
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

        # Start streaming
        profile = pipeline.start(config)
        print("RealSense stream started.")
        # Optional: Get device info
        device = profile.get_device()
        print(f"Connected to: {device.get_info(rs.camera_info.name)}")

    except RuntimeError as e:
        print(f"Error initializing RealSense: {e}")
        print("Check camera connection and librealsense installation.")
        return
    # --- RealSense Initialized ---

    # ORB detector
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    trained_objects = []
    quit_program = False

    try: # Wrap main loop in try/finally to ensure pipeline stops
        for obj_id in range(1, 4): # Train 3 objects
            if quit_program: break

            # Reset state for the new object selection
            refPt.clear()
            cropping = False
            selection_window_name = f"Select Object {obj_id}"
            cv2.namedWindow(selection_window_name)
            cv2.setMouseCallback(selection_window_name, click_and_crop)

            name = input(f"Enter name for object {obj_id}: ")
            print("\n--- Instructions ---\n1. Position object clearly.\n2. Draw a TIGHT bounding box.\n3. Press 's' to start capture.\n4. Press 'q' to quit.\n")

            # Selection Loop
            while True:
                frame = None
                try:
                    # --- Get frame from RealSense ---
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                    if not frames: continue
                    color_frame = frames.get_color_frame()
                    if not color_frame: continue
                    frame = np.asanyarray(color_frame.get_data())
                    # --- Frame Acquired ---
                except RuntimeError as e:
                     print(f"RealSense Error during selection: {e}")
                     time.sleep(0.1)
                     continue

                if frame is None: continue

                disp_select = draw_box(frame)
                cv2.putText(disp_select, "Draw box, then press 's'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(selection_window_name, disp_select)

                key = cv2.waitKey(1) & 0xFF # Essential for window updates
                if key == ord('s') and len(refPt) == 2:
                    print("Starting auto-capture...")
                    break
                elif key == ord('q'):
                    print("Quitting.")
                    quit_program = True
                    break # Exit selection loop

            cv2.destroyWindow(selection_window_name)
            if quit_program: break # Exit object loop if quitting

            # --- Capture descriptors with CSRT tracking ---
            print(f"Tracking '{name}'. Keep it in view. Move it closer/further to test scaling.")
            # Pass the pipeline object to the tracking function
            desc_list = auto_capture_tracked(pipeline, orb, num_samples=50, delay=CAPTURE_DELAY)

            if desc_list:
                valid_descriptors = [d for d in desc_list if d is not None and len(d) > 0]
                if valid_descriptors:
                    trained_objects.append({'id': obj_id, 'name': name, 'descriptors': valid_descriptors})
                    print(f"Captured {len(valid_descriptors)} valid samples for '{name}'.")
                else:
                    print(f"No valid descriptors captured for '{name}'. Skipping.")
            else:
                print(f"Capture failed or was interrupted for '{name}'. Skipping object.")

            if not quit_program: # Avoid pause if quitting
                 print("Prepare for next object...")
                 time.sleep(1) # Brief pause

        # --- End of Object Loop ---

    finally: # Ensure pipeline is stopped even if errors occur
        print("Stopping RealSense pipeline...")
        pipeline.stop()
        print("RealSense stopped.")
        cv2.destroyAllWindows() # Close any remaining OpenCV windows

    # Save data if objects were trained and we didn't quit early
    if trained_objects and not quit_program:
        try:
            with open('trainedObjects.pkl', 'wb') as f:
                pickle.dump(trained_objects, f)
            print("\nTraining complete. Data saved to trainedObjects.pkl.")
        except Exception as e:
            print(f"Error saving data to pickle file: {e}")
    elif not trained_objects:
        print("\nNo objects were successfully trained. No data saved.")
    else:
        print("\nTraining quit early. No data saved.")


if __name__ == '__main__':
    main()