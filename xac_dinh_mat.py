import cv2

# Mở video từ một file hoặc thiết bị như webcam
video_capture = cv2.VideoCapture(0)

# Tạo bộ phân loại khuôn mặt và mắt sử dụng pre-trained haarcascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')     # phần xác định mặt 
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    # Đọc một khung hình từ video
    ret, frame = video_capture.read()

    if not ret:
        break

    # Chuyển khung hình sang grayscale để tăng hiệu suất
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong khung hình
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))    # phần xác định mặt

    for (x, y, w, h) in faces:
        # Tính toán vị trí của nửa phía trên của khuôn mặt                                            # phần xác định mặt
        y_top = y
        y_bottom = y + h // 2
        x_left = x
        x_right = x + w

        # Cắt ảnh nửa phía trên của khuôn mặt                                                         # phần xác định mặt
        roi_face = frame[y_top:y_bottom, x_left:x_right]

        # Phát hiện mắt trong phần trên của khuôn mặt đã cắt
        roi_gray = gray[y_top:y_bottom, x_left:x_right]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Vẽ hình chữ nhật xung quanh mắt phát hiện được
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_face, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

        # Vẽ hình chữ nhật xung quanh khuôn mặt                                                      # phần xác định mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Hiển thị khung hình
    cv2.imshow('Face and Eye Detection', frame)

    # Thoát khỏi vòng lặp nếu người dùng nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng cửa sổ
video_capture.release()
cv2.destroyAllWindows()
