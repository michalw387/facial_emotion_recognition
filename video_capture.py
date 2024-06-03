import cv2
import face_recognition
import numpy as np


from detecting_emotions import get_emotion_prediction
from face_detection_from_photos import get_faces_from_images
import config
from utility_functions import apply_hog

# from model import EvaluateModel


def video_capture(model, hog=False, dim=3):

    video_capture = cv2.VideoCapture(0)

    frame_counter = 0

    face_locations, emotions = [], []

    font = cv2.FONT_HERSHEY_DUPLEX

    while True:
        frame_counter += 1
        _, current_frame = video_capture.read()

        if frame_counter % config.SKIP_FRAME_VIDEO_CAPTURE == 0:

            face_locations = face_recognition.face_locations(current_frame)

            if len(face_locations) == 0:
                continue

            faces = get_faces_from_images(current_frame, show_image=False, crop=True)

            emotions = []

            print(np.array(faces).shape)

            for face in faces[0]:

                face = apply_hog(face) if hog else face

                print(face.shape)
                emotion = get_emotion_prediction(model, face)
                emotions.append(emotion)
                print(f"Emotion: {emotion}")
                print("-----------")

        for (top, right, bottom, left), emotion in zip(face_locations, emotions):
            # Draw a box around the face
            cv2.rectangle(current_frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.putText(
                current_frame,
                emotion,
                (left + 6, bottom + 25),
                font,
                1.0,
                (255, 255, 255),
                1,
            )

        cv2.imshow("Video", current_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
