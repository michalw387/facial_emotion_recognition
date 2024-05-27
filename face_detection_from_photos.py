from PIL import Image
import dlib
from skimage import io
import face_recognition
import cv2
import numpy as np

VIDEOS_DIRECTORY = "Data\\Images\\"
IMAGE_FILES = ["barack.jpg", "baracks.jpg", "people.jpg"]

EMOTIONS_DICT = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}


def crop_faces_from_image(image, face_locations):
    face_images = []

    for face_location in face_locations:
        top, right, bottom, left = face_location

        face_image = image[top:bottom, left:right]

        face_images.append(face_image)

    return face_images


def show_image_with_rectangles(
    image, face_locations, label=None, print_location=False, bgr_image=False
):
    for face_location in face_locations:
        top, right, bottom, left = face_location

        # Draw rectangle around the face
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)

        if label:
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                image, label, (left + 6, bottom + 25), font, 1.0, (255, 255, 255), 1
            )

        if print_location:
            print(
                "A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(
                    top, left, bottom, right
                )
            )

    if bgr_image:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image with rectangles
    cv2.imshow(
        "Faces",
        image,
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_faces_from_images(
    images, show_image=False, labels_indexes=None, crop=False, only_one_face=False
):
    all_images_faces = []
    label = None

    for i, image in enumerate(images):
        image_faces_locations = get_faces_locations_from_image(image)

        if crop:
            cropped_faces = crop_faces_from_image(
                image, image_faces_locations
            )  # (faces, height, width, channels)

            if only_one_face:
                cropped_faces = cropped_faces[0]  # (height, width, channels)

            all_images_faces.append(cropped_faces)
        else:
            all_images_faces.append(image_faces_locations)

        if labels_indexes:
            label = EMOTIONS_DICT[labels_indexes[i]]

        if show_image:
            show_image_with_rectangles(image, image_faces_locations, label=label)

    return all_images_faces


def get_faces_locations_from_image(image):
    face_locations = face_recognition.face_locations(image)
    return face_locations


def get_faces_from_image_file(image_path):
    image = face_recognition.load_image_file(image_path)
    return get_faces_locations_from_image(image)


def find_faces_with_landmarks(file_name):

    predictor_model = "shape_predictor_68_face_landmarks.dat"

    # Create a HOG face detector using the built-in dlib class
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)

    win = dlib.image_window()

    # Load the image
    image = io.imread(file_name)

    # Run the HOG face detector on the image data
    detected_faces = face_detector(image, 1)

    print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

    # Show the desktop window with the image
    win.set_image(image)

    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):

        # Detected faces are returned as an object with the coordinates
        # of the top, left, right and bottom edges
        print(
            "- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(
                i,
                face_rect.left(),
                face_rect.top(),
                face_rect.right(),
                face_rect.bottom(),
            )
        )

        # Draw a box around each face we found
        win.add_overlay(face_rect)

        # Get the the face's pose
        pose_landmarks = face_pose_predictor(image, face_rect)

        # Draw the face landmarks on the screen.
        win.add_overlay(pose_landmarks)

    dlib.hit_enter_to_continue()


if __name__ == "__main__":
    for file_name in IMAGE_FILES:
        f = get_faces_from_image_file(VIDEOS_DIRECTORY + file_name)

    for file_name in IMAGE_FILES:
        find_faces_with_landmarks(VIDEOS_DIRECTORY + file_name)
