import os
import random
import cv2
from face_detection_from_photos import *
import numpy as np
import random

VIDEOS_DIRECTORY = "Data\\Videos\\"

FRAMES_PER_VIDEO = 1

MAX_VIDEOS_PER_PERSON = 10
MAX_FOLDERS = 3


def split_dataset(dataset):
    test_videos = dataset.pop(random.randint(0, len(dataset) - 1))

    train_videos = []

    for person in dataset:
        for frame, emotion in zip(person["faces"], person["emotions"]):
            train_videos.append({"face": frame, "emotion": emotion})

    random.shuffle(train_videos)

    return train_videos, test_videos


def get_videos(show_frames=False):
    print("Loading videos...")

    dataset = []
    for folder in os.listdir(VIDEOS_DIRECTORY):
        if len(dataset) >= MAX_FOLDERS:
            break

        selected_frames = get_one_person_videos(folder)

        frames = [frame["image"] for frame in selected_frames]
        emotions = [frame["emotion"] for frame in selected_frames]

        detected_faces = get_faces_from_images(
            frames,
            show_frames,
            labels_indexes=[frame["emotion"] for frame in selected_frames],
            crop=True,
            only_one_face=True,
        )

        dataset.append({"faces": detected_faces, "emotions": emotions})

    print("Videos Loaded:")
    print(
        f"People: {len(dataset)}, frames/labels: {len(dataset[0]['faces'])}, width0: {len(dataset[0]['faces'][0])},",
        end="",
    )
    print(
        f" height0: {len(dataset[0]['faces'][0][0])}, channels: {len(dataset[0]['faces'][0][0][0])}"
    )
    return dataset


def get_one_person_videos(folder, show_frames=False):
    selected_frames = []

    for video_filename in os.listdir(VIDEOS_DIRECTORY + folder):
        if len(selected_frames) >= MAX_VIDEOS_PER_PERSON:
            break
        emotion_index = int(video_filename[6:8])

        video_path = os.path.join(VIDEOS_DIRECTORY + folder, video_filename)

        cap = cv2.VideoCapture(video_path)

        all_frames = []

        while cap.isOpened():
            ret, frame = cap.read()  # frame size: (720, 1280, 3)
            if ret:
                all_frames.append(frame)
            else:
                break

        used_frames = []

        quarter_video_size = (len(all_frames) - 1) // FRAMES_PER_VIDEO

        for offset in range(FRAMES_PER_VIDEO):
            i = random.randint(
                quarter_video_size * offset + 10, quarter_video_size * (offset + 1) - 10
            )

            while i in used_frames:
                i = random.randint(0, len(all_frames) - 1)

            selected_frames.append({"image": all_frames[i], "emotion": emotion_index})

            used_frames.append(i)

        if show_frames:
            for frame in selected_frames[-4:]:
                get_faces_locations_from_image(frame, show_frames)

    return selected_frames


def save_to_file(dataset, filename="videos_dataset.npy"):
    print("Saving dataset to file...")
    np.save(filename, dataset)
    print("Dataset saved to file.")


def load_from_file(filename="videos_dataset.npy"):
    print("Loading dataset from file...")
    dataset = np.load(filename, allow_pickle=True)
    print("Dataset loaded from file.")
    return list(dataset)


# videos_dataset = get_videos(show_frames=False)

# save_to_file(videos_dataset)

videos_dataset = load_from_file()

train_set, test_set = split_dataset(videos_dataset)
