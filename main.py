from cnn_models import Model3D, Model2D
from model_evaluation import EvaluateModel
from video_capture import video_capture
from utility_functions import apply_hog, resize_image

import cv2

import config

if __name__ == "__main__":
    ev1 = EvaluateModel(model=Model3D())

    ev1.load_model("emotions_recognition_model_3D.pt")

    video_capture(ev1)

    ev2 = EvaluateModel(model=Model2D())

    ev2.load_model("hog_model_2D.pt")

    video_capture(ev2, hog=True)

    filenames = ["happy.jpg", "sad.jpg", "angry1.jpg", "angry2.jpg"]

    for file in filenames:

        img = cv2.imread(file)

        img = resize_image(img, 1000)

        apply_hog(img)

    #     img = resize_image(img, 1000)
    #     # img = Image.fromarray(np.uint8(img)).convert("RGB")
    #     # img.thumbnail((1000, 1000), PIL.Image.LANCZOS)
    #     # img = np.array(img)

    #     face = get_faces_from_images(
    #         img, show_image=False, crop=True, only_one_face=True
    #     )

    #     emotion = get_emotion_prediction(ev1, face)

    #     print(f"Emotion for {file}: {emotion}")
