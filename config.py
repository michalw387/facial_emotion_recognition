# File with all the configurations for the project

# files
VIDEOS_DIRECTORY = "Data\\Videos\\"
IMAGES_DIRECTORY = "Data\\Images\\"
MODELS_DIRECTORY = "Models\\"
DATASETS_DIRECTORY = "Data\\Datasets\\"

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
MAX_FOLDERS = None
MAX_VIDEOS_PER_PERSON = None
FRAMES_PER_VIDEO = 4
IMAGE_SQUARE_SIZE = 200

# face detection
SKIP_FRAME_VIDEO_CAPTURE = 2

# filters
APPLY_HOG = False

# model
BATCH_SIZE = 100
VALIDATION_SET_SIZE = 0.1
