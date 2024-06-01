import numpy as np
import cv2
import PIL
from PIL import Image
import face_recognition

import config


def upscale_image_to_desire_size(img, size=config.IMAGE_SQUARE_SIZE):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.uint8(img)).convert("RGB")
    if img.size[0] < size and img.size[1] < size:
        if img.size[0] > img.size[1]:
            mywidth = size
            wpercent = mywidth / float(img.size[0])
            myheight = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((mywidth, myheight), PIL.Image.LANCZOS)
        else:
            myheight = size
            hpercent = myheight / float(img.size[1])
            mywidth = int((float(img.size[0]) * float(hpercent)))
            img = img.resize((mywidth, myheight), PIL.Image.LANCZOS)

    return np.array(img)


def expand_image_to_square(img, background_color=(255, 255, 255)):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.uint8(img)).convert("RGB")
    width, height = img.size
    if width == height:
        return img
    elif width > height:
        result = Image.new(img.mode, (width, width), background_color)
        result.paste(img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(img.mode, (height, height), background_color)
        result.paste(img, ((height - width) // 2, 0))
        return result


def resize_image(PIL_image):
    if not isinstance(PIL_image, Image.Image):
        PIL_image = Image.fromarray(np.uint8(PIL_image)).convert("RGB")

    desire_size = (config.IMAGE_SQUARE_SIZE, config.IMAGE_SQUARE_SIZE)

    PIL_image.thumbnail(desire_size, PIL.Image.LANCZOS)
    PIL_image = upscale_image_to_desire_size(PIL_image, config.IMAGE_SQUARE_SIZE)
    PIL_image = expand_image_to_square(PIL_image)

    return np.array(PIL_image)


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
    images,
    show_image=False,
    labels_indexes=None,
    crop=False,
    only_one_face=False,
):
    all_images_faces = []
    label = None

    for i, image in enumerate(images):
        image_faces_locations = get_faces_locations_from_image(image)

        if crop:
            cropped_faces = crop_faces_from_image(
                image, image_faces_locations
            )  # (faces, height, width, channels)

            resizes_faces = []

            for face in cropped_faces:
                square_face = resize_image(face)
                # dodać transformację twarzy do landmarków
                resizes_faces.append(square_face)

            cropped_faces = np.array(resizes_faces)

            if only_one_face:
                cropped_faces = cropped_faces[0]  # (height, width, channels)

            all_images_faces.append(cropped_faces)
        else:
            all_images_faces.append(image_faces_locations)

        if labels_indexes:
            label = config.EMOTIONS_DICT[labels_indexes[i]]

        if show_image:
            show_image_with_rectangles(image, image_faces_locations, label=label)

    return all_images_faces


def get_faces_locations_from_image(image):
    return face_recognition.face_locations(image)


def get_faces_from_image_file(image_path):
    image = face_recognition.load_image_file(image_path)
    return get_faces_locations_from_image(image)


# def find_faces_with_landmarks(file_name):

#     predictor_model = "shape_predictor_68_face_landmarks.dat"

#     # Create a HOG face detector using the built-in dlib class
#     face_detector = dlib.get_frontal_face_detector()
#     face_pose_predictor = dlib.shape_predictor(predictor_model)

#     win = dlib.image_window()

#     # Load the image
#     image = io.imread(file_name)

#     # Run the HOG face detector on the image data
#     detected_faces = face_detector(image, 1)

#     print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

#     # Show the desktop window with the image
#     win.set_image(image)

#     # Loop through each face we found in the image
#     for i, face_rect in enumerate(detected_faces):

#         # Detected faces are returned as an object with the coordinates
#         # of the top, left, right and bottom edges
#         print(
#             "- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(
#                 i,
#                 face_rect.left(),
#                 face_rect.top(),
#                 face_rect.right(),
#                 face_rect.bottom(),
#             )
#         )

#         # Draw a box around each face we found
#         win.add_overlay(face_rect)

#         # Get the the face's pose
#         pose_landmarks = face_pose_predictor(image, face_rect)

#         # Draw the face landmarks on the screen.
#         win.add_overlay(pose_landmarks)

#     dlib.hit_enter_to_continue()


if __name__ == "__main__":
    for file_name in config.IMAGE_FILES:
        faces_locations = get_faces_from_image_file(config.VIDEOS_DIRECTORY + file_name)

        print("--------------------")
        print(np.array(faces_locations))
        for loc in faces_locations:
            top, right, bottom, left = loc
            print(f"width: {right-left}, height: {bottom-top}")

        show_image_with_rectangles(
            cv2.imread(config.VIDEOS_DIRECTORY + file_name), faces_locations
        )

    # for file_name in config.IMAGE_FILES:
    #     find_faces_with_landmarks(config.VIDEOS_DIRECTORY + file_name)
