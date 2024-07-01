import numpy as np
import torch

from image_processing import ImageProcessing


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

    while face.ndim < 4:
        face = face.unsqueeze(0)

    return face


def get_emotion_prediction(model, face):
    face = normalize_to_tensor(face)

    tensor_emotion = model.predict(face)

    tensor_emotion = torch.tensor(
        ImageProcessing.readjust_indexes_emotions(tensor_emotion.cpu())
    ).float()

    emotion = model.get_emotion_from_tensor(tensor_emotion)

    return emotion
