from PIL import Image
import PIL
import numpy as np
import torch


def normalize_image(image):
    image = np.array(image)
    image = image / 255
    return image


def normalize_to_tensor(face, transpose=True):
    face = normalize_image(face)

    face = torch.tensor(np.array(face, dtype=np.float32))

    face.squeeze_(0)

    if transpose:
        face = face.T

    face = face.unsqueeze(0)

    return face


def get_emotion_prediction(model, face):
    face = normalize_to_tensor(face)

    tensor_emotion = model.predict(face)

    emotion = model.get_emotion_from_tensor(tensor_emotion)

    return emotion


# import random

# EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


# def get_emotion(face):
#     return random.choice(EMOTIONS)


# def detect_emotion(frame, face_locations):
#     emotions = []
#     for top, right, bottom, left in face_locations:
#         face = frame[top:bottom, left:right]
#         emotion = get_emotion(face)
#         emotions.append(emotion)

#     return emotions
