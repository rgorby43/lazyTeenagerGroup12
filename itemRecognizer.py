import cv2
import pickle

# Sensitivity parameters
DIST_THRESH = 40    # Hamming distance threshold for a "good" match
COUNT_THRESH = 20    # Minimum match count to confidently label

# Load trained objects from pickle
with open('trainedObjects.pkl', 'rb') as f:
    trained_objects = pickle.load(f)

# Initialize ORB detector and matcher
orb = cv2.ORB_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Open webcam (or change index for other camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit(1)

print("Starting recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray, None)

    best_match = None
    best_count = 0

    if des_frame is not None:
        # Compare against each trained object
        for obj in trained_objects:
            total_matches = 0
            for des in obj['descriptors']:
                # Use KNN matching for more candidates
                matches = matcher.knnMatch(des, des_frame, k=2)
                # Lowe's ratio test
                good = [m for m, n in matches if m.distance < DIST_THRESH * 0.8 and m.distance < 0.75 * n.distance]
                total_matches += len(good)
            # Track best
            if total_matches > best_count:
                best_count = total_matches
                best_match = obj['name']

    # Build label with confidence feedback
    if best_match:
        if best_count >= COUNT_THRESH:
            label = f"{best_match} ({best_count})"
            color = (0, 255, 0)
        else:
            label = f"Maybe: {best_match} ({best_count})"
            color = (0, 165, 255)  # orange for low confidence
    else:
        label = "None"
        color = (0, 0, 255)

    cv2.putText(frame, f"Detected: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
