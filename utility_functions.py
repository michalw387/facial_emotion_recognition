import numpy as np
import PIL
from PIL import Image
from skimage.feature import hog
from skimage import exposure


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
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True,
            channel_axis=-1 if frame.ndim == 3 else None,
        )

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        hog_frames.append(hog_image_rescaled)

    return np.array(hog_frames)


def resize_image(PIL_image, max_size):
    if not isinstance(PIL_image, Image.Image):
        PIL_image = Image.fromarray(np.uint8(PIL_image)).convert("RGB")

    PIL_image.thumbnail((max_size, max_size), PIL.Image.LANCZOS)

    return np.array(PIL_image)
