import numpy as np
import dlib
import cv2

from skimage import io
from image_processing import ImageProcessing


class EyeMouthImages:
    @staticmethod
    def min_max_points(points):
        min_x = min(points, key=lambda p: p.x).x
        min_y = min(points, key=lambda p: p.y).y
        max_x = max(points, key=lambda p: p.x).x
        max_y = max(points, key=lambda p: p.y).y
        return (min_x, min_y), (max_x, max_y)

    @staticmethod
    def get_eye_points(pose_landmarks):
        right_eye_points = pose_landmarks.parts()[36:42]
        return right_eye_points

    @staticmethod
    def get_mouth_points(pose_landmarks):
        mouth_points = pose_landmarks.parts()[48:68]
        return mouth_points

    @staticmethod
    def get_eye_mouth_images(face_image, face_rect=None):
        predictor_model = "shape_predictor_68_face_landmarks.dat"
        face_pose_predictor = dlib.shape_predictor(predictor_model)

        if isinstance(face_image, list):
            face_image = np.array(face_image)

        if face_image.dtype != "uint8":
            face_image = face_image.astype("uint8")

        height, width = face_image.shape[:2]

        if face_rect is None:
            face_rect = dlib.rectangle(0, 0, width, height)

        pose_landmarks = face_pose_predictor(face_image, face_rect)

        right_eye_points = EyeMouthImages.get_eye_points(pose_landmarks)
        mouth_points = EyeMouthImages.get_mouth_points(pose_landmarks)

        (eye_min_x, eye_min_y), (eye_max_x, eye_max_y) = EyeMouthImages.min_max_points(
            right_eye_points
        )
        (mouth_min_x, mouth_min_y), (mouth_max_x, mouth_max_y) = (
            EyeMouthImages.min_max_points(mouth_points)
        )

        mouth_x_offset, mouth_y_offset = 10, 10
        eye_x_offset, eye_y_offset = 15, 15

        eye_min_y = max(eye_min_y - eye_y_offset, 0)
        eye_max_y = min(eye_max_y + eye_y_offset, height)
        eye_min_x = max(eye_min_x - eye_x_offset, 0)
        eye_max_x = min(eye_max_x + eye_x_offset, width)

        mouth_min_y = max(mouth_min_y - mouth_y_offset, 0)
        mouth_max_y = min(mouth_max_y + mouth_y_offset, height)
        mouth_min_x = max(mouth_min_x - mouth_x_offset, 0)
        mouth_max_x = min(mouth_max_x + mouth_x_offset, width)

        right_eye_image = face_image[eye_min_y:eye_max_y, eye_min_x:eye_max_x]
        mouth_image = face_image[mouth_min_y:mouth_max_y, mouth_min_x:mouth_max_x]

        if right_eye_image.size == 0:
            right_eye_image = face_image
        if mouth_image.size == 0:
            mouth_image = face_image

        return right_eye_image, mouth_image

    @staticmethod
    def convert_dlib_rectangles_to_tuples(rectangles):
        return [
            (rect.top(), rect.right(), rect.bottom(), rect.left())
            for rect in rectangles
        ]


if __name__ == "__main__":
    # Load image
    image = io.imread("Data\\Images\\angry1.jpg")

    # Convert image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = ImageProcessing.resize_image(image, 1000)

    image = np.array(image)

    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()

    # Resize the image if it's too large
    image = ImageProcessing.resize_image(image, 1000)

    # Run the HOG face detector on the image data
    detected_faces = face_detector(image)

    detected_faces = EyeMouthImages.convert_dlib_rectangles_to_tuples(detected_faces)

    cropped_faces = ImageProcessing.crop_faces_from_image(
        image, detected_faces
    )  # (faces, height, width, channels)

    resized_faces = []

    for face in cropped_faces:
        square_face = ImageProcessing.resize_image_to_square(face)
        # dodać transformację twarzy do landmarków
        resized_faces.append(square_face)

    resized_faces[0] = cv2.cvtColor(resized_faces[0], cv2.COLOR_RGB2GRAY)

    # mouth_image, right_eye_image = Landmarks.get_eye_mouth_images(image, detected_faces[0])
    mouth_image, right_eye_image = EyeMouthImages.get_eye_mouth_images(resized_faces[0])
    cv2.imshow("Image", resized_faces[0])
    cv2.imshow("Mouth", mouth_image)
    cv2.imshow("Right eye", right_eye_image)

    # frame = find_faces_with_landmarks_cv(image)
    # cv2.imshow("Video", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
