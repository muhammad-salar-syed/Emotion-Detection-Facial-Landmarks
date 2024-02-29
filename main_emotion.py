import cv2
import mediapipe as mp
import os
import numpy as np


def get_face_landmarks(image, draw=False, static_image_mode=True):

    # Read the input image
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=static_image_mode,
                                                max_num_faces=1,
                                                min_detection_confidence=0.5)
    image_rows, image_cols, _ = image.shape
    results = face_mesh.process(image_input_rgb)

    image_landmarks = []

    if results.multi_face_landmarks:

        if draw:

            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

        ls_single_face = results.multi_face_landmarks[0].landmark
        xs_ = []
        ys_ = []
        zs_ = []
        for idx in ls_single_face:
            xs_.append(idx.x)
            ys_.append(idx.y)
            zs_.append(idx.z)
        for j in range(len(xs_)):
            image_landmarks.append(xs_[j] - min(xs_))
            image_landmarks.append(ys_[j] - min(ys_))
            image_landmarks.append(zs_[j] - min(zs_))

    return image_landmarks


data_dir = './faces_final'

output = []
for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
    for image_path_ in os.listdir(os.path.join(data_dir, emotion)):
        image_path = os.path.join(data_dir, emotion, image_path_)

        image = cv2.imread(image_path)

        face_landmarks = get_face_landmarks(image)
        #print(len(face_landmarks))

        if len(face_landmarks) == 1404:
            face_landmarks.append(int(emotion_indx))
            output.append(face_landmarks)

np.savetxt('data_final.txt', np.asarray(output))


###################################################################
## train model

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data from the text file
data_file = "data_final.txt"
data = np.loadtxt(data_file)

# Split data into features (X) and labels (y)
X = data[:, :-1]  # Features are all columns except the last one
y = data[:, -1]   # Labels are the last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    shuffle=True,
                                                    stratify=y)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(confusion_matrix(y_test, y_pred))

with open('./model_final', 'wb') as f:
    pickle.dump(rf_classifier, f)