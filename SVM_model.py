import cv2
import numpy as np
from sklearn import svm
import os

        

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




# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Create a VideoCapture object to capture video from the webcam
cap = cv2.VideoCapture(0)  # You can specify the webcam index (0 for the default camera)





# Load your SVM model and data capturing functions here
def flip_horizontal(image):
    return cv2.flip(image, 1)

def capture_data(eye_open_dir, eye_closed_dir):
    X_eye_open, y_eye_open = [], []
    for filename in os.listdir(eye_open_dir):
        img = cv2.imread(os.path.join(eye_open_dir, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50, 50))
        feature = np.sum(img > 100) / (img.size)
        X_eye_open.append(feature)
        y_eye_open.append(1)  # Nhãn 1 cho mắt mở
    
    X_eye_closed, y_eye_closed = [], []
    for filename in os.listdir(eye_closed_dir):
        img = cv2.imread(os.path.join(eye_closed_dir, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50, 50))
        feature = np.sum(img > 100) / (img.size)
        X_eye_closed.append(feature)
        y_eye_closed.append(0)  # Nhãn 0 cho mắt nhắm
    
    X = np.array(X_eye_open + X_eye_closed).reshape(-1, 1)
    y = np.array(y_eye_open + y_eye_closed)

    return X, y

def train_svm(X, y):
    clf = svm.SVC(C=1.0, kernel='linear')
    clf.fit(X, y)
    return clf

# Khai báo các đường dẫn cho mắt mở và mắt nhắm
eye_open_dir = "Data_Open"
eye_closed_dir = "Data_Closed"

X, y = capture_data(eye_open_dir, eye_closed_dir)
clf = train_svm(X, y)

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

eye_status = "Unknown"
close_time = 0







while True:
    ret, frame = cap.read()
    frame = flip_horizontal(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not ret:
        break

    frame_id += 1

    # Face detection
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

                    # Determine eye status here for the tracked face
                    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
                    if len(eyes) == 0:
                        close_time += 1
                        new_face['eye_status'] = "eye_close"
                    else:
                        close_time = 0
                        new_face['eye_status'] = "eye_open"
                        
                    found_match = True
                    break

        if not found_match:
            
            # If no matching tracked face is found, consider it a new face and add it to the list
            new_face = {'centroid': face_centroid, 'width': rect[2], 'id': len(new_tracked_faces) + 1,'eye_status': "Unknown"}
            # Determine eye status for the new face here
            
            new_tracked_faces.append(new_face)


    # Update the tracked_faces list with the new list of tracked faces
    tracked_faces = new_tracked_faces 

    # Draw bounding boxes, IDs, and eye status on the frame for each tracked face
    for tracked_face in tracked_faces:
        x, y = tracked_face['centroid']
        face_id = tracked_face['id']
        eye_status = tracked_face['eye_status']
        cv2.rectangle(frame, (x - 50, y - 50), (x + 50, y + 50), (0, 255, 0), 2)
        cv2.putText(frame, f'ID:{face_id},{eye_status}', (x - 90, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        

    cv2.imshow('phat_hien_mat_tap_trung', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



