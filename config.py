# File with all the configurations for the project

# Files
VIDEOS_DIRECTORY = "Data\\Videos\\"
IMAGES_DIRECTORY = "Data\\Images\\"
MODELS_DIRECTORY = "Models\\"
DATASETS_DIRECTORY = "Data\\Datasets\\"

IMAGE_FILES = ["barack.jpg", "baracks.jpg", "people.jpg"]

# Emotions
EMOTIONS_DICT = {
    1: "neutral",
    3: "happy",
    4: "sad",
    5: "angry",
}
NUM_EMOTIONS = len(EMOTIONS_DICT)

# Videos
MAX_FOLDERS = None
MAX_VIDEOS_PER_PERSON = None
FRAMES_PER_VIDEO = 8
IMAGE_SQUARE_SIZE = 100

# Face detection
SKIP_FRAME_VIDEO_CAPTURE = 2

# Filters
APPLY_HOG = False
APPLY_LOG = False
APPLY_LANDMARKS = False
GENERATE_MOUTH_AND_EYE = False

# Model
BATCH_SIZE = 100
VALIDATION_SET_SIZE = 0.1
