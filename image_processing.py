import numpy as np
import cv2
from skimage.feature import hog
from PIL import Image

import config


class ImageProcessing:

    @staticmethod
    def adjust_indexes_emotions(emotions):  # przełożyć do detecting_emotions.py
        adjusted_emotions = []

        for emotion in emotions:
            if emotion == 1:
                adjusted_emotions.append(emotion - 1)
            else:
                adjusted_emotions.append(emotion - 2)

        return np.array(adjusted_emotions)

    @staticmethod
    def readjust_indexes_emotions(emotions):  # przełożyć do detecting_emotions.py
        adjusted_emotions = []

        for emotion in emotions:
            if emotion == 0:
                adjusted_emotions.append(emotion + 1)
            else:
                adjusted_emotions.append(emotion + 2)

        return np.array(adjusted_emotions)

    @staticmethod
    def apply_hog(frames):
        hog_frames = []
        frames = np.array(frames)

        if frames.ndim == 3:
            frames = [frames]

        for frame in frames:
            frame = np.array(frame)

            # Compute HOG features and the corresponding HOG image
            _, hog_image = hog(
                frame,
                orientations=8,
                pixels_per_cell=(16, 16),
                cells_per_block=(1, 1),
                visualize=True,
                channel_axis=-1 if frame.ndim == 3 else None,
            )

            cv2.imshow("HOG Image", hog_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            hog_frames.append(hog_image)

        return np.array(hog_frames)

    @staticmethod
    def resize_image(PIL_image, max_size):
        if not isinstance(PIL_image, Image.Image):
            PIL_image = Image.fromarray(np.uint8(PIL_image)).convert("RGB")

        PIL_image.thumbnail((max_size, max_size), Image.LANCZOS)

        return np.array(PIL_image)

    @staticmethod
    def upscale_image_to_desired_size(img, size):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.uint8(img)).convert("RGB")

        if img.size[0] < size and img.size[1] < size:
            if img.size[0] > img.size[1]:
                mywidth = size
                wpercent = mywidth / float(img.size[0])
                myheight = int((float(img.size[1]) * float(wpercent)))
                img = img.resize((mywidth, myheight), Image.LANCZOS)
            else:
                myheight = size
                hpercent = myheight / float(img.size[1])
                mywidth = int((float(img.size[0]) * float(hpercent)))
                img = img.resize((mywidth, myheight), Image.LANCZOS)
        return np.array(img)

    @staticmethod
    def expand_image_to_square(img, background_color=(255, 255, 255)):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.uint8(img)).convert("RGB")
        width, height = img.size
        if width == height:
            return np.array(img)
        elif width > height:
            square_img = Image.new(img.mode, (width, width), background_color)
            square_img.paste(img, (0, (width - height) // 2))
            return np.array(square_img)
        else:
            square_img = Image.new(img.mode, (height, height), background_color)
            square_img.paste(img, ((height - width) // 2, 0))
            return np.array(square_img)

    @staticmethod
    def resize_image_to_square(PIL_image, size=None, two_dim=False):
        if not isinstance(PIL_image, Image.Image):
            PIL_image = Image.fromarray(np.uint8(PIL_image)).convert("RGB")

        if size is None:
            size = config.IMAGE_SQUARE_SIZE

        desired_size = (size, size)

        PIL_image.thumbnail(desired_size, Image.LANCZOS)
        np_image = ImageProcessing.upscale_image_to_desired_size(PIL_image, size)
        np_image = ImageProcessing.expand_image_to_square(np_image)

        if two_dim:
            np_image = np_image[:, :, 0]

        return np.array(np_image)

    @staticmethod
    def crop_faces_from_image(image, face_locations):
        face_images = []

        for face_location in face_locations:
            top, right, bottom, left = face_location

            face_image = image[top:bottom, left:right]

            face_images.append(face_image)

        return face_images
