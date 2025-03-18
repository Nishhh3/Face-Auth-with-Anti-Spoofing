import cv2
import dlib
import numpy as np
import time
from scipy.spatial import distance as dist

# Load pre-trained face and landmark detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file

# Eye aspect ratio (EAR) calculation function
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

# Eye landmark indexes (based on dlib's 68 facial keypoints)
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# EAR Thresholds
EAR_THRESHOLD = 0.25  # Below this value indicates blink
CONSECUTIVE_FRAMES = 2  # Number of frames required to confirm a blink

# Anti-spoofing parameters
SPOOF_TIMEOUT = 5.0  # Time in seconds to wait for a blink before declaring a spoof
blink_counter = 0
blink_detected = False
spoof_detected = False
face_detected = False
start_time = time.time()

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot access webcam")
        break

    # Create a copy for display
    display_frame = frame.copy()
    
    # Convert frame to grayscale (for faster processing)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    
    # Reset face_detected flag
    face_detected = False

    for face in faces:
        face_detected = True
        landmarks = predictor(gray, face)
        
        # Extract eye coordinates
        left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE])
        right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE])

        # Draw eye regions
        hull_left = cv2.convexHull(left_eye)
        hull_right = cv2.convexHull(right_eye)
        cv2.drawContours(display_frame, [hull_left], -1, (0, 255, 0), 1)
        cv2.drawContours(display_frame, [hull_right], -1, (0, 255, 0), 1)

        # Compute EAR for both eyes
        left_EAR = calculate_ear(left_eye)
        right_EAR = calculate_ear(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0
        
        # Display current EAR value
        cv2.putText(display_frame, f"EAR: {avg_EAR:.2f}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Check if blink is detected
        if avg_EAR < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= CONSECUTIVE_FRAMES:
                blink_detected = True
                spoof_detected = False
                # Reset the timer when a blink is detected
                start_time = time.time()
            blink_counter = 0  # Reset counter

        # Draw bounding box around the face
        cv2.rectangle(display_frame, (face.left(), face.top()), 
                      (face.right(), face.bottom()), (0, 255, 0), 2)

    # If no face is detected, reset the timer and flags
    if not face_detected:
        start_time = time.time()
        blink_detected = False
        spoof_detected = False
        cv2.putText(display_frame, "No Face Detected", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # Check if we've waited too long without a blink
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Display countdown timer
        remaining_time = max(0, SPOOF_TIMEOUT - elapsed_time)
        cv2.putText(display_frame, f"Time to blink: {remaining_time:.1f}s", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if not blink_detected and elapsed_time > SPOOF_TIMEOUT:
            spoof_detected = True

        # Display anti-spoofing status
        if spoof_detected:
            status_text = "SPOOF DETECTED! "
            color = (0, 0, 255)  # Red for spoof
        elif blink_detected:
            status_text = "REAL PERSON "
            color = (0, 255, 0)  # Green for real person
        else:
            status_text = "Waiting for blink..."
            color = (255, 165, 0)  # Orange for waiting
            
        cv2.putText(display_frame, status_text, (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Add instructions
    cv2.putText(display_frame, "Press 'q' to quit, 'r' to reset", 
                (20, display_frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
    # Show the frame
    cv2.imshow("Blink Detection Anti-Spoofing System", display_frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset the system
        start_time = time.time()
        blink_detected = False
        spoof_detected = False

cap.release()
cv2.destroyAllWindows()