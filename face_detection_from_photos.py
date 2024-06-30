import numpy as np
import cv2
import face_recognition
from tqdm import tqdm

import config
from image_processing import ImageProcessing


def show_image_with_rectangles(
    image, face_locations, label=None, print_location=False, bgr_image=False
):
    for face_location in face_locations:
        top, right, bottom, left = face_location

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

    cv2.imshow(
        "Faces",
        image,
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_faces_from_images(
    images,
    show_image=False,
    labels_indexes=None,
    crop=False,
    only_one_face=False,
    folder=None,
    show_tqdm=True,
):
    all_images_faces = []

    if images.ndim == 3:
        images = np.array([images])

    for i, image in enumerate(tqdm(images, disable=(not show_tqdm))):

        txt_folder = f"from folder {folder}" if folder else ""

        if show_tqdm:
            tqdm.write(f"Processing image {i + 1}/{len(images)} {txt_folder}")

        image_faces_locations = get_faces_locations_from_image(image)

        no_faces_condition = len(image_faces_locations) == 0

        if no_faces_condition:
            continue

        if crop:
            cropped_faces = ImageProcessing.crop_faces_from_image(
                image, image_faces_locations
            )  # (faces, height, width, channels)

            resizes_faces = []

            for face in cropped_faces:
                square_face = ImageProcessing.resize_image_to_square(face)
                resizes_faces.append(square_face)

            cropped_faces = np.array(resizes_faces)

            if only_one_face:
                cropped_faces = cropped_faces[0]  # (height, width, channels)

            all_images_faces.append(cropped_faces)
        else:
            all_images_faces.append(image_faces_locations)

        if show_image:
            label = config.EMOTIONS_DICT[labels_indexes[i]] if labels_indexes else None
            show_image_with_rectangles(image, image_faces_locations, label=label)

    return all_images_faces


def get_faces_locations_from_image(image):
    return face_recognition.face_locations(image)


def get_faces_from_image_file(image_path):
    image = face_recognition.load_image_file(image_path)
    return get_faces_locations_from_image(image)
