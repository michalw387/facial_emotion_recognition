import os
import random
import cv2
from face_detection_from_photos import *
import numpy as np
import random

VIDEOS_DIRECTORY = "Data\\Videos\\"

FRAMES_PER_VIDEO = 1

MAX_VIDEOS_PER_PERSON = 5
MAX_FOLDERS = 3


def split_dataset(dataset, dataset_labels):
    print("Splitting dataset...")

    i = random.randint(0, len(dataset) - 1)

    print(np.array(dataset_labels).shape)

    test_videos = dataset.pop(i)
    test_labels = dataset_labels.pop(i)

    # dataset = np.array(dataset)

    # print(dataset)

    # print("-----------------------")
    # print("-----------------------")
    # print("-----------------------")
    # print("-----------------------")

    # flattened_dataset = []

    # for person in dataset:
    #     for frame in person["faces"]:
    #         flattened_dataset.append(frame)

    # new_dataset = dataset.reshape(-1, 2, 321, 321, 3)

    # print(new_dataset)

    print(np.array(dataset_labels).shape)

    dataset = np.array(dataset)
    dataset_labels = np.array(dataset_labels)

    combined = list(zip(dataset.reshape(-1, 321, 321, 3), dataset_labels.reshape(-1)))

    # Pomieszanie zgrupowanych danych
    random.shuffle(combined)

    print("combined", np.array(combined).shape)

    # Rozpakowanie danych i etykiet z powrotem do tablic
    shuffled_data, shuffled_labels = zip(*combined)

    train_size = int(len(dataset) * 0.8)

    train_set = dataset[:train_size]
    test_set = dataset[train_size:]

    return train_set, test_set


def get_videos(show_frames=False):
    print("Loading videos...")

    dataset = []
    dataset_labels = []
    for folder in os.listdir(VIDEOS_DIRECTORY):
        if len(dataset) >= MAX_FOLDERS:
            break

        selected_frames, frames_emotions = get_one_person_videos(folder)

        # frames = [frame["image"] for frame in selected_frames]
        # emotions = [frame["emotion"] for frame in selected_frames]

        frames = selected_frames
        emotions = frames_emotions

        detected_faces = get_faces_from_images(
            frames,
            show_frames,
            labels_indexes=emotions,
            crop=True,
            only_one_face=True,
        )

        dataset.append(detected_faces)
        dataset_labels.append(emotions)

    # print(
    #     f"persons:{len(dataset)} frames/labels:{len(dataset[0]['faces'])} width:{len(dataset[0]['faces'][0])}",
    #     end="",
    # )
    # print(
    #     f" height:{len(dataset[0]['faces'][0][0])} channels:{len(dataset[0]['faces'][0][0][0])}"
    # )
    return dataset, dataset_labels


def get_one_person_videos(folder, show_frames=False):
    selected_frames = []
    frames_emotions = []

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

            selected_frames.append(all_frames[i])
            frames_emotions.append(emotion_index)

            used_frames.append(i)

        if show_frames:
            for frame in selected_frames[-4:]:
                get_faces_locations_from_image(frame, show_frames)

    return selected_frames, frames_emotions


dataset, dataset_labels = get_videos(show_frames=False)

train_set, test_set = split_dataset(dataset, dataset_labels)
