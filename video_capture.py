import cv2
import face_recognition
from detecting_emotions import detect_emotion


# from model import EvaluateModel


SKIP_FRAME = 2

video_capture = cv2.VideoCapture(0)

frame_counter = 0

face_locations, emotions = [], []

# ev1 = EvaluateModel()
# ev1.start()

while True:
    frame_counter += 1
    ret, frame = video_capture.read()

    name = "Unknown"

    if frame_counter % SKIP_FRAME == 0:
        face_locations = face_recognition.face_locations(frame)

        emotions = detect_emotion(frame, face_locations)

    for (top, right, bottom, left), emotion in zip(face_locations, emotions):
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(
        #     frame, emotion, (left + 6, bottom + 25), font, 1.0, (255, 255, 255), 1
        # )

    # Display the resulting image
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
