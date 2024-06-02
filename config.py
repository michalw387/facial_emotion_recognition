# File with all the configurations for the project

# files
VIDEOS_DIRECTORY = "Data\\Videos\\"
IMAGES_DIRECTORY = "Data\\Images\\"

IMAGE_FILES = ["barack.jpg", "baracks.jpg", "people.jpg"]

# emotions
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
NUM_EMOTIONS = len(EMOTIONS_DICT)

# videos
FRAMES_PER_VIDEO = 1
MAX_VIDEOS_PER_PERSON = None
MAX_FOLDERS = None
IMAGE_SQUARE_SIZE = 100
