from PIL import Image
import numpy as np
import PIL
import cv2

from face_detection_from_photos import (
    upscale_image_to_desire_size,
    expand_image_to_square,
    get_faces_from_images,
)


filename = "test.jpg"
filename = "test2.jpg"

get_faces_from_images(cv2.imread(filename), show_image=True, crop=True)
# filename = "small_test.jpg"
# filename = "small_test2.jpg"

SIZE = 800
IMAGE_SIZE = (SIZE, SIZE)


img = cv2.imread(filename)
print(img.shape)

cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


resized_img = cv2.resize(img, IMAGE_SIZE)
print(resized_img.shape)

cv2.imshow("test", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


PIL_image = Image.fromarray(np.uint8(img)).convert("RGB")
PIL_image.thumbnail(IMAGE_SIZE, PIL.Image.LANCZOS)
PIL_image = upscale_image_to_desire_size(PIL_image, SIZE)
print(np.array(PIL_image).shape)
cv2.imshow("test", np.array(PIL_image))
cv2.waitKey(0)
cv2.destroyAllWindows()
PIL_image = expand_image_to_square(PIL_image)

print(np.array(PIL_image).shape)
cv2.imshow("test", np.array(PIL_image))
cv2.waitKey(0)
cv2.destroyAllWindows()

# resized_PIL_image = PIL_image.resize(image_size)
# print(np.array(resized_PIL_image).shape)

# cv2.imshow("test", np.array(resized_PIL_image))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# mywidth = SIZE

# img = Image.open(filename)
# wpercent = mywidth / float(img.size[0])
# hsize = int((float(img.size[1]) * float(wpercent)))
# img = img.resize((mywidth, hsize), PIL.Image.LANCZOS)
# img = img.convert("RGB")

# cv2.imshow("test", np.array(img))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
