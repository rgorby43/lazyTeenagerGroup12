import cv2
import pickle
import numpy as np
import time

# Globals for mouse callback
refPt = []
cropping = False
mx, my = -1, -1  # mouse position during drag


def click_and_crop(event, x, y, flags, param):
    """
    Mouse callback: record bounding box start/end and track dragging.
    """
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


def auto_capture_tracked(cap, orb, num_samples=30, delay=0.5):
    """
    Automatically capture samples while tracking ROI via a CSRT tracker.
    Returns list of descriptor arrays.
    """
    ret, frame = cap.read()
    while not ret:
        print("Waiting for initial frame...")
        time.sleep(0.1)
        ret, frame = cap.read()

    if len(refPt) != 2:
        print("Error: Bounding box not selected correctly.")
        return []

    (x1, y1), (x2, y2) = refPt
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        print("Error: Invalid bounding box dimensions.")
        return []

    # Check OpenCV version for tracker API compatibility
    major_ver, minor_ver, _ = cv2.__version__.split('.')
    if int(major_ver) < 4 or (int(major_ver) == 4 and int(minor_ver) < 5):
        try:
            tracker = cv2.legacy.TrackerCSRT_create()
        except AttributeError:
             print("Error: Legacy TrackerCSRT not found. Ensure opencv-contrib-python is installed.")
             return []
    else:
        try:
            tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            print("Error: TrackerCSRT not found. Check OpenCV installation.")
            return []

    try:
        tracker.init(frame, (x1, y1, w, h))
        print(f"Tracker initialized with box: x={x1}, y={y1}, w={w}, h={h}")
    except Exception as e:
        print(f"Error initializing tracker: {e}")
        return []


    descriptors = []
    captured_count = 0
    frame_count = 0
    # Set a max attempts limit for frame processing
    max_attempts = num_samples * 5

    while captured_count < num_samples and frame_count < max_attempts:
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            print("Warning: Could not read frame from camera.")
            time.sleep(0.1)
            continue

        ok, bbox = tracker.update(frame)
        disp = frame.copy()

        if ok:
            x, y, w, h = map(int, bbox)

            # Ensure ROI coordinates are within frame boundaries
            if w > 0 and h > 0 and x >= 0 and y >= 0 and (x + w) <= frame.shape[1] and (y + h) <= frame.shape[0]:
                roi = frame[y:y+h, x:x+w]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                kp, des = orb.detectAndCompute(gray_roi, None)

                if des is not None:
                    descriptors.append(des)
                    captured_count += 1
                    cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 0), 2) # Green box: success

                    # Offset keypoint coordinates relative to the full frame for display
                    kp_disp = []
                    for p in kp:
                        # Create a new KeyPoint with adjusted coordinates
                        kp_disp.append(cv2.KeyPoint(x=p.pt[0] + x, y=p.pt[1] + y, size=p.size,
                                                   angle=p.angle, response=p.response, octave=p.octave,
                                                   class_id=p.class_id))
                    cv2.drawKeypoints(disp, kp_disp, disp, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                else:
                    cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 255), 2) # Yellow box: no features
                    cv2.putText(disp, "No features", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 0, 255), 2) # Red box: invalid ROI dims
                cv2.putText(disp, "Invalid ROI dims", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(f"Warning: Invalid ROI dimensions - x={x}, y={y}, w={w}, h={h}, frame_w={frame.shape[1]}, frame_h={frame.shape[0]}")

        else: # Tracking failed
            print(f"Tracking failure on frame {frame_count}, captured {captured_count}/{num_samples}")
            cv2.putText(disp, "Tracking Failure", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(disp, f"{captured_count}/{num_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Training - Auto Capture", disp)

        if cv2.waitKey(max(1, int(delay * 1000))) & 0xFF == ord('q'):
            print("Capture interrupted by user.")
            break

    if captured_count < num_samples:
        print(f"Warning: Only captured {captured_count} out of {num_samples} desired samples.")

    # Destroy only the auto-capture window
    cv2.destroyWindow("Training - Auto Capture")
    return descriptors


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    # ORB detector with potentially more features
    orb = cv2.ORB_create(nfeatures=1500)
    trained_objects = []

    for obj_id in range(1, 4):
        # Reset state for the new object selection
        refPt.clear()
        cropping = False
        selection_window_name = f"Select Object {obj_id}"
        cv2.namedWindow(selection_window_name)
        cv2.setMouseCallback(selection_window_name, click_and_crop)

        name = input(f"Enter name for object {obj_id}: ")
        print("\n--- Instructions ---\n1. Position object clearly.\n2. Draw a TIGHT bounding box.\n3. Press 's' to start capture.\n4. Press 'q' to quit.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            disp_select = draw_box(frame)
            cv2.putText(disp_select, "Draw box, then press 's'", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(selection_window_name, disp_select)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and len(refPt) == 2:
                print("Starting auto-capture...")
                break
            elif key == ord('q'):
                print("Quitting.")
                cap.release()
                cv2.destroyAllWindows()
                return

        # Close the selection window
        cv2.destroyWindow(selection_window_name)

        print(f"Tracking '{name}'. Keep it in view. Move it closer/further to test scaling.")
        desc_list = auto_capture_tracked(cap, orb, num_samples=100, delay=0.1)

        # Add object data only if descriptors were successfully captured
        if desc_list:
             # Filter out any None or empty descriptor arrays
            valid_descriptors = [d for d in desc_list if d is not None and len(d) > 0]
            if valid_descriptors:
                trained_objects.append({'id': obj_id, 'name': name, 'descriptors': valid_descriptors})
                print(f"Captured {len(valid_descriptors)} valid samples for '{name}'.")
            else:
                print(f"No valid descriptors captured for '{name}'. Skipping.")
        else:
            print(f"Capture failed or was interrupted for '{name}'. Skipping object.")

        print("Prepare for next object...")
        time.sleep(1)

    if trained_objects:
        try:
            with open('trainedObjects.pkl', 'wb') as f:
                pickle.dump(trained_objects, f)
            print("\nTraining complete. Data saved to trainedObjects.pkl.")
        except Exception as e:
            print(f"Error saving data to pickle file: {e}")
    else:
        print("\nNo objects were successfully trained. No data saved.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()