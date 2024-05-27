import random

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def get_emotion(face):
    return random.choice(EMOTIONS)


def detect_emotion(frame, face_locations):
    emotions = []
    for top, right, bottom, left in face_locations:
        face = frame[top:bottom, left:right]
        emotion = get_emotion(face)
        emotions.append(emotion)

    return emotions
