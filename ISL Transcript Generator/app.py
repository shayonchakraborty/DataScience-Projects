import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('Action_latest.keras')

# Initialize MediaPipe components
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Define actions and their specific thresholds
actions = ['hello', 'thanks', 'iloveyou']
thresholds = {'hello': 0.90, 'thanks': 0.8, 'iloveyou': 0.60 }

def mediapipe_detection(image, holistic):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,250), thickness=2, circle_radius=2))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Streamlit UI
st.title('Real-Time Gesture Recognition')
st.sidebar.header('Settings')
st.sidebar.subheader('Parameters')

# Create placeholders for video feed and predictions
video_placeholder = st.empty()
predictions_placeholder = st.empty()

# Initialize Streamlit session state
if 'video_active' not in st.session_state:
    st.session_state.video_active = False
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = ""
if 'all_predictions' not in st.session_state:
    st.session_state.all_predictions = []

# Start Video button
if st.sidebar.button('Start Video'):
    st.session_state.video_active = True

if st.session_state.video_active:
    cap = cv2.VideoCapture(0)
    sequence = []
    sentence = []
    predictions = []

    # Initialize MediaPipe Holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while st.session_state.video_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Error accessing the webcam.")
                break

            # Process frame
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            # Extract keypoints and make predictions
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                max_prob = np.max(res)
                current_prediction = actions[np.argmax(res)]
                
                # Check if the prediction exceeds the specific threshold for that action
                if (current_prediction == 'hello' and max_prob <= thresholds['hello']) or (current_prediction != 'hello' and max_prob > thresholds[current_prediction]):
                    if current_prediction != st.session_state.last_prediction:
                        st.session_state.last_prediction = current_prediction
                        st.session_state.all_predictions.append(current_prediction)

            # Convert image to format suitable for Streamlit
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            video_placeholder.image(image_rgb, channels="RGB", use_column_width=True)

            # Display all combined predictions, but only display if there are new predictions
            if st.session_state.all_predictions:
                predictions_placeholder.text(f"Combined Actions: {' '.join(st.session_state.all_predictions)}")

            # Stop capturing if the user closes the browser tab or stops interaction
            if not st.session_state.video_active:
                break

    cap.release()

# Additional buttons
st.sidebar.button('Transcript')
st.sidebar.button('Voice')




































