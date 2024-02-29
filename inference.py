import cv2
import mediapipe as mp
import pickle
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
            blank_img = img = img = np.zeros(image.shape, np.uint8)
            mp_drawing = mp.solutions.drawing_utils
            #mp_drawing_styles = mp.solutions.drawing_styles
            drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

            mp_drawing.draw_landmarks(
                image=blank_img,
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

    return image_landmarks, blank_img



emotions = ['happy', 'sad']

with open('./model_final', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture('./demo.mp4')
ret, frame = cap.read()
#print(frame.shape)
H, W, _ = frame.shape
out = cv2.VideoWriter('./out.mp4', cv2.VideoWriter_fourcc(*'mpv4'), int(cap.get(cv2.CAP_PROP_FPS)), (W*2, H))

while ret:
    #ret, frame = cap.read()
    face_landmarks, blank_img = get_face_landmarks(frame, draw=True, static_image_mode=False)

    output = model.predict([face_landmarks])

    cv2.putText(blank_img,emotions[int(output[0])], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2)
    
    im_h = cv2.hconcat([frame,blank_img]) 
    cv2.imshow('frame',im_h)
    #print(im_h.shape)

    cv2.waitKey(1)
    out.write(im_h)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()



    
