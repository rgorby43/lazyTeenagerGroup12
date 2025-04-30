# face_detector_module.py

import cv2
import numpy as np
import time
import pyrealsense2 as rs
import os
import threading

class RealSenseFaceDetector:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None
        self.face_cascade = None
        self._is_running = False # Internal flag for loop control
        self._detection_thread = None
        self._detection_successful = False # Flag to store the outcome
        self._required_duration = 1 # Default, can be overridden
        self.window_name = "Face Detection - Waiting..."

        print("RealSenseFaceDetector initialized for waiting.")

    # --- Initialization Methods (_initialize_cascade, _initialize_realsense) ---
    # Keep these exactly the same as in the previous version
    def _initialize_cascade(self):
        # ... (same code as before) ...
        haar_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if not os.path.exists(haar_cascade_path):
            print(f"Error: Haar Cascade file not found at {haar_cascade_path}")
            return False
        try:
            self.face_cascade = cv2.CascadeClassifier(haar_cascade_path)
            if self.face_cascade.empty():
                print(f"Error: Failed to load Haar Cascade classifier from {haar_cascade_path}")
                return False
            print("Haar Cascade face detector loaded successfully.")
            return True
        except Exception as e:
            print(f"Error initializing Cascade Classifier: {e}")
            return False

    def _initialize_realsense(self):
        # ... (same code as before) ...
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                print("No RealSense devices connected.")
                return False

            print(f"Configuring RealSense stream: {self.width}x{self.height} @ {self.fps} FPS")
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

            profile = self.pipeline.start(config)
            print("RealSense stream started.")
            device = profile.get_device()
            print(f"Connected to: {device.get_info(rs.camera_info.name)}")
            return True

        except RuntimeError as e:
            print(f"Error initializing RealSense: {e}")
            self.pipeline = None
            return False
        except Exception as e:
            print(f"An unexpected error occurred during RealSense init: {e}")
            self.pipeline = None
            return False
    # --- End Initialization Methods ---


    def _run_detection_loop_for_wait(self):
        """Internal loop that runs until consistent face or manual stop."""
        print("DEBUG: MODULE - Detection loop (wait mode) thread started.")
        self._detection_successful = False # Ensure reset before loop
        face_detected_start_time = None # Track consistency start time

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            print(f"DEBUG: MODULE - Window created. Waiting for {self._required_duration}s consistency...")
        except Exception as e:
            print(f"DEBUG: MODULE - FAILED TO CREATE WINDOW: {e}")
            self._is_running = False # Stop if window fails
            return # Exit thread

        while self._is_running:
            frame = None
            current_time = time.time()

            # --- Get frame ---
            if not self.pipeline:
                print("Error: RealSense pipeline not available in loop.")
                self._is_running = False
                break
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                if not frames:
                    time.sleep(0.01)
                    continue
                color_frame = frames.get_color_frame()
                if not color_frame:
                    time.sleep(0.01)
                    continue
                frame = np.asanyarray(color_frame.get_data())
            except RuntimeError as e:
                print(f"RealSense Error during frame acquisition: {e}")
                time.sleep(0.2)
                continue # Attempt to recover
            except Exception as e:
                 print(f"Unexpected error getting frame: {e}")
                 self._is_running = False # Stop on unexpected errors
                 break

            if frame is None: continue

            # --- Face Detection ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            face_present = len(faces) > 0

            # --- Consistency Check ---
            if face_present:
                if face_detected_start_time is None:
                    # Start timer on first detection in a sequence
                    face_detected_start_time = current_time
                    print(f"DEBUG: MODULE - Face detected, starting consistency timer ({current_time:.2f})...")
                else:
                    # Check if duration met
                    elapsed = current_time - face_detected_start_time
                    if elapsed >= self._required_duration:
                        print(f"DEBUG: MODULE - Consistent face detected for {elapsed:.2f}s. Success!")
                        self._detection_successful = True
                        self._is_running = False # Signal loop to stop
                        # No need to break here, loop condition will handle it
            else:
                if face_detected_start_time is not None:
                    print("DEBUG: MODULE - Face lost, resetting consistency timer.")
                # Reset timer if face is lost
                face_detected_start_time = None
            # --- End Consistency Check ---

            # --- Prepare Display ---
            status_text = "Looking for face..."
            color = (0, 165, 255) # Orange
            if face_detected_start_time is not None:
                 elapsed_str = f"{(current_time - face_detected_start_time):.1f}s / {self._required_duration}s"
                 status_text = f"Face detected! Holding... {elapsed_str}"
                 color = (0, 255, 255) # Yellow
            if face_present:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Blue rectangle
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # Optional FPS
            # ... (FPS calculation can be added back if desired) ...

            # --- Display Frame ---
            try:
                 cv2.imshow(self.window_name, frame)
            except Exception as e:
                  print(f"DEBUG: MODULE - cv2.imshow FAILED: {e}")
                  self._is_running = False # Stop if display fails
                  # No need to break, loop condition handles it

            # --- Handle Window Events & Manual Exit ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("DEBUG: MODULE - 'q' pressed, stopping detection.")
                self._detection_successful = False # Ensure success is false on manual quit
                self._is_running = False
                # No need to break, loop condition handles it

            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("DEBUG: MODULE - Window closed manually, stopping detection.")
                    self._detection_successful = False # Ensure success is false
                    self._is_running = False
                    # No need to break, loop condition handles it
            except cv2.error:
                if self._is_running: # Only print error if we weren't already stopping
                     print("DEBUG: MODULE - Window property check failed (likely closing), stopping loop.")
                     self._is_running = False
            except Exception as e:
                 print(f"Unexpected error checking window property: {e}")
                 self._is_running = False # Stop on unexpected errors


        # --- Loop exited ---
        print("DEBUG: MODULE - Detection loop (wait mode) finished.")
        # Explicitly destroy window here after loop finishes
        try:
             cv2.destroyWindow(self.window_name)
             cv2.waitKey(1) # Short wait seems necessary sometimes after destroy
        except Exception as e:
             print(f"DEBUG: MODULE - Error destroying window (may already be closed): {e}")

    def wait_for_consistent_face(self, duration=0.5):
        """
        Starts detection, shows window, and waits until a face is
        detected consistently for 'duration' seconds.

        Args:
            duration (float): How many seconds a face must be detected
                              continuously. Defaults to 0.5.

        Returns:
            bool: True if a face was detected consistently, False otherwise
                  (e.g., manual stop via 'q' or closing window).
        """
        print(f"Waiting for face detection (consistency: {duration}s)...")
        self._detection_successful = False # Reset status
        self._is_running = False
        self._required_duration = duration # Store duration for the loop

        # --- Initialize ---
        if not self._initialize_cascade():
            print("Failed to initialize cascade. Cannot start detection.")
            return False
        if not self._initialize_realsense():
            print("Failed to initialize RealSense. Cannot start detection.")
            self._cleanup_resources() # Clean up cascade if RS failed
            return False

        # --- Start and Wait for Thread ---
        self._is_running = True
        # Run the loop in a thread, but make it non-daemon so we can reliably join
        self._detection_thread = threading.Thread(target=self._run_detection_loop_for_wait)
        self._detection_thread.start()
        print("DEBUG: MODULE - Detection thread started, joining...")

        self._detection_thread.join() # Wait here until the thread finishes
        print("DEBUG: MODULE - Detection thread joined.")

        # --- Cleanup ---
        self._cleanup_resources()

        # --- Return Result ---
        print(f"Detection finished. Success status: {self._detection_successful}")
        return self._detection_successful

    def _cleanup_resources(self):
        """Stops pipeline and cleans up OpenCV windows."""
        print("DEBUG: MODULE - Cleaning up resources...")
        if self.pipeline:
            try:
                active_profile = self.pipeline.get_active_profile()
                if active_profile:
                    print("Stopping RealSense pipeline...")
                    self.pipeline.stop()
                    print("RealSense pipeline stopped.")
                else:
                    print("RealSense pipeline was not active or already stopped.")
            except Exception as e:
                print(f"Error stopping RealSense (may be normal if already stopped): {e}")
            finally:
                self.pipeline = None # Ensure reset

        # Ensure all windows are closed, even if thread had issues
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass # Ignore errors during final cleanup

    def __del__(self):
        """Destructor attempts cleanup."""
        print("RealSenseFaceDetector destructor called.")
        # Destructor less critical now as the main method blocks and cleans up,
        # but good practice to keep a basic version.
        self._is_running = False # Signal any lingering thread (unlikely now)
        self._cleanup_resources()