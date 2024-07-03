import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./sample_model.p', 'rb'))
model = model_dict['model']

# Initialize video capture (try with index 0)
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

while True:
    data_aux = []
    x_ = []
    y_ = []

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is captured successfully
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

                # Normalize and add coordinates
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Pad data_aux if its length is less than 84
        while len(data_aux) < 84:
            data_aux.append(0.0)

        # Ensure data_aux is exactly 84 features
        data_aux = data_aux[:84]

        # Check the shape of data_aux
        print(f"data_aux shape: {np.asarray(data_aux).shape}")

        prediction = model.predict([np.asarray(data_aux)])

        print(f"Prediction: {prediction}")

        try:
            predicted_character = prediction[0]
        except IndexError as e:
            predicted_character = "Unknown"
            print(f"Error in prediction: {e}")

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
