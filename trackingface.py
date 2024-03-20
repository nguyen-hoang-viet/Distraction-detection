import cv2
import numpy as np

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a VideoCapture object to capture video from the webcam or a video file
cap = cv2.VideoCapture("4.mp4")

# Create an initial empty list to store information about detected faces
tracked_faces = []

# Define a function to perform mean-shift tracking
def mean_shift_tracking(frame, target_features, initial_centroid):
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    x, y, w, h = initial_centroid[0] - 25, initial_centroid[1] - 25, 50, 50
    track_window = (x, y, w, h)
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    while True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        x, y, w, h = track_window
        return (x, y, w, h)
    
# Define a function to calculate the centroid of a bounding box
def calculate_centroid(rect):
    x, y, w, h = rect
    cx = x + w // 2
    cy = y + h // 2
    return (cx, cy)

# Define a function to calculate the distance between two centroids
def calculate_distance(centroid1, centroid2):
    return np.linalg.norm(np.array(centroid1) - np.array(centroid2))

# Define a function to extract features from a face (e.g., using a pre-trained model)
def extract_face_features(frame, rect):
    x, y, w, h = rect
    face = frame[y:y+h, x:x+w]
    # Implement feature extraction (e.g., using a deep learning model)
    # Extract and return the features as a numpy array
    return np.array([])  # Replace with actual feature extraction code

frame_id = 0

# Set thresholds for distance and size
distance_threshold = 80
size_threshold = 50

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # Update the list of tracked faces
    new_tracked_faces = []

    for rect in faces:
        face_centroid = calculate_centroid(rect)
        found_match = False

        for tracked_face in tracked_faces:
            # Calculate the Euclidean distance between the centroid of the new face
            # and the centroid of the tracked face
            dist = np.linalg.norm(np.array(face_centroid) - np.array(tracked_face['centroid']))

            if dist < tracked_face['width'] // 2:
                # If the distance is smaller than half of the width of the tracked face,
                # consider it the same face, and update its centroid
                # Extract face features and use them for tracking
                tracked_face_features = extract_face_features(frame, rect)
                if tracked_face_features is not None:
                    new_rect = mean_shift_tracking(frame, tracked_face_features, tracked_face['centroid'])
                    tracked_face['centroid'] = calculate_centroid(new_rect)
                    tracked_face['width'] = new_rect[2]
                    new_tracked_faces.append(tracked_face)
                found_match = True
                break

        if not found_match:
            # If no matching tracked face is found, consider it a new face and add it to the list
            new_tracked_faces.append({'centroid': face_centroid, 'width': rect[2], 'id': len(new_tracked_faces) + 1})

    # Update the tracked_faces list with the new list of tracked faces
    tracked_faces = new_tracked_faces

    # Draw bounding boxes and IDs on the frame
    for tracked_face in tracked_faces:
        x, y = tracked_face['centroid']
        face_id = tracked_face['id']
        cv2.rectangle(frame, (x - 50, y - 50), (x + 50, y + 50), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {face_id}', (x - 40, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()


