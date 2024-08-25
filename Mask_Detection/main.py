from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from mtcnn import MTCNN

# Initialize MTCNN face detector
face_detector = MTCNN()

# Load the mask detection model
try:
    classifier = load_model(r'C:\programming\DataScience\DataScience-Projects\Mask_Detection\mask_detector.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

classes = ['No Mask', 'Mask']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # Convert the frame to RGB ( MTCNN works with RGB images )
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using MTCNN
    faces = face_detector.detect_faces(rgb_frame)

    for face in faces:
        x, y, w, h = face['box']
        confidence = face['confidence']

        if confidence > 0.9:  # Confidence threshold
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

            # Extract the region of interest (ROI) and preprocess it for prediction
            roi_rgb = rgb_frame[y:y+h, x:x+w]
            roi_rgb = cv2.resize(roi_rgb, (128, 128), interpolation=cv2.INTER_AREA)
            roi_rgb = roi_rgb.astype('float32') / 255.0
            roi_rgb = img_to_array(roi_rgb)
            roi_rgb = np.expand_dims(roi_rgb, axis=0)

            # Predict mask/no-mask
            prediction = classifier.predict(roi_rgb)[0]
            label = classes[prediction.argmax()]
            # label_position = (x, y - 10)
            # cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(label)

    cv2.imshow('Mask Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
