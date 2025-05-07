# face_detector_module.py
import sys
sys.path.append('/home/group12/')

import cv2
import numpy as np
import time
import pyrealsense2 as rs
import os
# import threading # No longer needed for this revised approach

class RealSenseFaceDetector:
    def __init__(self, width=640, height=480, fps=30, external_pipeline=None):
        print(f"RFS_FD __init__: Instance {id(self)} creation started.")
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = external_pipeline
        self._using_external_pipeline = (external_pipeline is not None)

        if self._using_external_pipeline:
            print(f"RFS_FD __init__: Instance {id(self)} using EXTERNAL pipeline id {id(self.pipeline)}.")
        else:
            print(f"RFS_FD __init__: Instance {id(self)} will manage INTERNAL pipeline (initially None).")

        self.face_cascade = None
        # self._is_running = False # Not needed in the same way without a separate thread
        # self._detection_thread = None # Not needed
        self._detection_successful = False
        self._required_duration = 1 # Default, will be set by wait_for_consistent_face
        self.window_name = "Face Detection - Waiting..." # Default window name
        print(f"RFS_FD __init__: Instance {id(self)} basic attributes set.")

    def _initialize_cascade(self):
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
        print(f"RFS_FD _initialize_realsense: Called for instance {id(self)}.")
        if self._using_external_pipeline and self.pipeline:
            print(f"RFS_FD _initialize_realsense: Using external pipeline id {id(self.pipeline)}.")
            try:
                # Check if the pipeline is active. If not, the caller must start it.
                # It's generally better if the main script fully manages the external pipeline's lifecycle.
                if self.pipeline.get_active_profile():
                    print("RFS_FD _initialize_realsense: External pipeline is already active.")
                else:
                    print("RFS_FD _initialize_realsense: External pipeline provided but NOT active. Caller must ensure it's started before use.")
                return True
            except RuntimeError as e:
                 print(f"RFS_FD _initialize_realsense: Error checking external pipeline status (likely not started or invalid): {e}. Assuming caller manages.")
                 return True # Still return true, as it's an external pipeline
            except Exception as e:
                print(f"RFS_FD _initialize_realsense: Unexpected error with external pipeline: {e}.")
                return False


        print(f"RFS_FD _initialize_realsense: Initializing internal pipeline for instance {id(self)}.")
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            # Note: Checking for devices like this is good, but rs.pipeline() might fail before this if no device.
            ctx = rs.context()
            devices = ctx.query_devices()
            if not devices: # Simplified check
                print("RFS_FD _initialize_realsense: No RealSense devices connected (internal init).")
                self.pipeline = None # Ensure pipeline is None
                return False
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.pipeline.start(config)
            print(f"RFS_FD _initialize_realsense: Internal pipeline id {id(self.pipeline)} started.")
            return True
        except RuntimeError as e: # Specifically catch RealSense runtime errors
            print(f"RFS_FD _initialize_realsense: RealSense RuntimeError during internal init for instance {id(self)}: {e}")
            if self.pipeline:
                try: self.pipeline.stop() # Attempt to stop if partially started
                except: pass
            self.pipeline = None
            return False
        except Exception as e:
            print(f"RFS_FD _initialize_realsense: FAILED for instance {id(self)} with unexpected error: {e}")
            if self.pipeline:
                try: self.pipeline.stop()
                except: pass
            self.pipeline = None
            return False

    def wait_for_consistent_face(self, duration=0.5, display_window_name="Face Detection - Waiting..."):
        """
        Shows window and waits until a face is detected consistently for 'duration' seconds.
        This is a BLOCKING call and manages its own OpenCV window.

        Args:
            duration (float): How many seconds a face must be detected continuously.
            display_window_name (str): Name for the OpenCV window.

        Returns:
            bool: True if a face was detected consistently, False otherwise.
        """
        print(f"RFS_FD wait_for_consistent_face: Waiting for face (consistency: {duration}s)...")
        self._detection_successful = False
        self._required_duration = duration
        self.window_name = display_window_name # Use provided or default

        # --- Initialize ---
        if not self._initialize_cascade():
            print("RFS_FD wait_for_consistent_face: Failed to initialize cascade.")
            return False # Cannot proceed
        if not self._initialize_realsense(): # This will check/use external or init internal
            print("RFS_FD wait_for_consistent_face: Failed to initialize RealSense.")
            self._cleanup_resources() # Clean up cascade if RS failed
            return False

        if not self.pipeline:
            print("RFS_FD wait_for_consistent_face: RealSense pipeline is not available after initialization. Aborting.")
            self._cleanup_resources()
            return False
        try:
            # Ensure the pipeline is active, especially if it's external and might not have been started
            if not self.pipeline.get_active_profile():
                if self._using_external_pipeline:
                    print("RFS_FD wait_for_consistent_face: ERROR - External pipeline is not active. The main script must start it.")
                    self._cleanup_resources()
                    return False
                else: # Should not happen if internal init was successful
                    print("RFS_FD wait_for_consistent_face: ERROR - Internal pipeline not active despite successful init.")
                    self._cleanup_resources()
                    return False
        except RuntimeError as e:
            print(f"RFS_FD wait_for_consistent_face: RuntimeError checking pipeline activity: {e}. Aborting.")
            self._cleanup_resources()
            return False


        # --- Direct Detection Loop (formerly _run_detection_loop_for_wait) ---
        print(f"RFS_FD wait_for_consistent_face: Starting detection loop in current thread for window '{self.window_name}'.")
        face_detected_start_time = None
        loop_active = True # Controls the loop

        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            print(f"RFS_FD wait_for_consistent_face: Window '{self.window_name}' created. Waiting for {self._required_duration}s consistency...")
        except Exception as e:
            print(f"RFS_FD wait_for_consistent_face: FAILED TO CREATE WINDOW '{self.window_name}': {e}")
            self._cleanup_resources()
            return False # Exit if window fails

        while loop_active:
            frame = None
            current_time = time.time()

            if not self.pipeline:
                print("RFS_FD wait_for_consistent_face: Error - RealSense pipeline became unavailable in loop.")
                loop_active = False
                break
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000) # ms
                if not frames:
                    time.sleep(0.01) # Brief pause if no frames
                    continue
                color_frame = frames.get_color_frame()
                if not color_frame:
                    time.sleep(0.01)
                    continue
                frame = np.asanyarray(color_frame.get_data())
            except RuntimeError as e:
                print(f"RFS_FD wait_for_consistent_face: RealSense Error during frame acquisition: {e}")
                # Decide if this is fatal or skippable
                if "Frame didn't arrive within" in str(e):
                    print("RFS_FD wait_for_consistent_face: Timeout waiting for frame, continuing...")
                    time.sleep(0.1) # Longer pause on timeout
                    continue
                else: # More serious RS error
                    loop_active = False # Stop loop on other RS errors
                    break
            except Exception as e:
                 print(f"RFS_FD wait_for_consistent_face: Unexpected error getting frame: {e}")
                 loop_active = False
                 break

            if frame is None: continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            face_present = len(faces) > 0

            if face_present:
                if face_detected_start_time is None:
                    face_detected_start_time = current_time
                    print(f"RFS_FD wait_for_consistent_face: Face detected, starting consistency timer ({current_time:.2f})...")
                else:
                    elapsed = current_time - face_detected_start_time
                    if elapsed >= self._required_duration:
                        print(f"RFS_FD wait_for_consistent_face: Consistent face detected for {elapsed:.2f}s. Success!")
                        self._detection_successful = True
                        loop_active = False # Signal loop to stop
            else:
                if face_detected_start_time is not None:
                    print("RFS_FD wait_for_consistent_face: Face lost, resetting consistency timer.")
                face_detected_start_time = None

            status_text = "Looking for face..."
            color = (0, 165, 255)
            if face_detected_start_time is not None:
                 elapsed_str = f"{(current_time - face_detected_start_time):.1f}s / {self._required_duration}s"
                 status_text = f"Face detected! Holding... {elapsed_str}"
                 color = (0, 255, 255)
            if face_present:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            try:
                 cv2.imshow(self.window_name, frame)
            except Exception as e:
                  print(f"RFS_FD wait_for_consistent_face: cv2.imshow FAILED for window '{self.window_name}': {e}")
                  loop_active = False # Stop if display fails

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"RFS_FD wait_for_consistent_face: 'q' pressed in '{self.window_name}', stopping detection.")
                self._detection_successful = False
                loop_active = False

            try:
                # Check if window was closed by user via 'X' button
                # WND_PROP_VISIBLE is >= 1.0 if visible. < 1.0 if not.
                # For some OpenCV versions/backends, 0 might mean closed.
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print(f"RFS_FD wait_for_consistent_face: Window '{self.window_name}' closed by user, stopping.")
                    self._detection_successful = False
                    loop_active = False
            except cv2.error: # cv2.error can be raised if window no longer exists
                if loop_active: # Only an issue if we weren't already stopping
                    print(f"RFS_FD wait_for_consistent_face: Window '{self.window_name}' property check failed (likely closed).")
                    loop_active = False # Assume closed, stop loop
            except Exception as e:
                 print(f"RFS_FD wait_for_consistent_face: Unexpected error checking window property: {e}")
                 loop_active = False # Stop on unexpected errors

        # --- Loop exited ---
        print(f"RFS_FD wait_for_consistent_face: Detection loop for '{self.window_name}' finished.")
        try:
             cv2.destroyWindow(self.window_name)
             # It's good practice to call waitKey(1) immediately after destroyWindow
             # to allow OpenCV to process the window destruction event.
             cv2.waitKey(1)
             print(f"RFS_FD wait_for_consistent_face: Window '{self.window_name}' destroyed.")
        except Exception as e:
             print(f"RFS_FD wait_for_consistent_face: Error destroying window '{self.window_name}' (may already be closed): {e}")

        self._cleanup_resources() # Perform resource cleanup (e.g., stop internal pipeline)
        print(f"RFS_FD wait_for_consistent_face: Detection finished. Success status: {self._detection_successful}")
        return self._detection_successful

    def _cleanup_resources(self):
        print(f"RFS_FD _cleanup_resources: Called for instance {id(self)}.")
        if not self._using_external_pipeline and self.pipeline:
            print(f"RFS_FD _cleanup_resources: Attempting to stop internally managed pipeline id {id(self.pipeline)}.")
            try:
                if self.pipeline.get_active_profile(): # Check if it's running before stopping
                    self.pipeline.stop()
                    print("RFS_FD _cleanup_resources: Internally managed pipeline stopped.")
                else:
                    print("RFS_FD _cleanup_resources: Internally managed pipeline was not active, no stop needed.")
            except RuntimeError as e:
                print(f"RFS_FD _cleanup_resources: Error stopping internal pipeline (may be normal if already stopped or unconfigured): {e}")
            # except Exception as e: # Catch any other potential errors
            #     print(f"RFS_FD _cleanup_resources: Unexpected error stopping internal pipeline: {e}")
            finally:
                self.pipeline = None
        elif self._using_external_pipeline:
            print(f"RFS_FD _cleanup_resources: Was using external pipeline id {id(self.pipeline)}. Not stopping it here.")
            # self.pipeline reference will be cleared when the instance is garbage collected, or can be set to None.
            # However, if the main script is still using this pipeline instance, we should not None it here.
            # The key is not to *stop* it.
        # self.face_cascade = None # Cascade doesn't need explicit release like pipeline

    def __del__(self):
        print(f"RFS_FD __del__: Destructor called for instance {id(self)}.")
        # _cleanup_resources now handles pipeline stopping correctly based on internal/external
        self._cleanup_resources()
        # It's generally good practice to try and destroy any windows this instance *might* have created,
        # though the main method (wait_for_consistent_face) should handle its own window.
        # This is more of a fallback.
        try:
            if hasattr(self, 'window_name') and self.window_name: # Check if window_name attribute exists
                # Check if window actually exists before trying to destroy
                # This is tricky as getWindowProperty might error if it doesn't exist.
                # A simple destroy attempt with a catch is often sufficient here.
                cv2.destroyWindow(self.window_name)
                cv2.waitKey(1)
        except Exception:
            pass # Suppress errors during destruction