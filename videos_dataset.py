import os
import random
import numpy as np
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray

import config
from face_detection_from_photos import get_faces_from_images
from utility_functions import apply_hog


class VideosDataset(Dataset):
    def __init__(self, frames, emotions):
        self.frames = torch.tensor(frames)
        self.emotions = torch.tensor(emotions)
        self.n_frames = len(self.frames)

    def __len__(self):
        return self.n_frames

    def __getitem__(self, index):
        return self.frames[index], self.emotions[index]


class LoadVideosDataset:
    def __init__(
        self,
        batch_size=config.BATCH_SIZE,
        val_set_size=config.VALIDATION_SET_SIZE,
        shuffle=True,
    ):
        self.dataset = None
        self.batch_size = batch_size
        self.val_set_size = val_set_size
        self.shuffle = shuffle

    def print_dataset_item(self, index=0):
        print("Emotions:", self.dataset[index]["emotions"])
        print("Frames:", self.dataset[index]["faces"])

    def print_dataset_info(self):
        print("--------Dataset info--------")
        print("People in dataset:", len(self.dataset))
        print("Frames per person:", len(self.dataset[0]["faces"]))
        print("Frame size:", np.array(self.dataset[0]["faces"][0]).shape)
        different_emotions = set(
            np.array(
                [self.dataset[i]["emotions"] for i in range(len(self.dataset))]
            ).flatten()
        )
        print(
            "Emotions in dataset:",
            [config.EMOTIONS_DICT[i] for i in different_emotions],
        )
        print("______________________________")

    def normalize_dataset(self):
        for person in self.dataset:
            for i in range(len(person["faces"])):
                person["faces"][i] = person["faces"][i] / 255

    def apply_face_filters(self, faces, gray_scale=False, transpose=True):
        faces = np.array(
            faces,
            dtype=np.float32,
        )

        if config.APPLY_HOG:
            faces = apply_hog(faces)
            # LOG ????????????????
        elif gray_scale:
            faces = np.array([rgb2gray(frame) for frame in faces])

        if transpose:  # ??????????????????????????????????????
            faces = np.array([frame.T for frame in faces])

        return np.array(faces)

    def get_test_data_loader(self, gray_scale=False, transpose=True):
        # test_frames = self.get_test_set()

        dataset_len = len(self.dataset) * len(self.dataset[0]["faces"])

        test_set_len = 0
        test_frames = {"faces": np.array(), "emotions": np.array()}

        while test_set_len < dataset_len * 0.1:
            frames = self.get_test_set()
            test_frames["faces"] = np.append(test_frames["faces"], frames["faces"])
            test_frames["emotions"] = np.append(
                test_frames["emotions"], frames["emotions"]
            )
            test_set_len = len(test_frames["faces"])

        test_emotions = test_frames["emotions"]

        test_faces = self.apply_face_filters(
            test_frames["faces"], gray_scale, transpose
        )

        # frames = np.array(
        #     test_frames["faces"],
        #     dtype=np.float32,
        # )

        # if config.APPLY_HOG:
        #     frames = apply_hog(frames)
        # elif gray_scale:
        #     frames = np.array([rgb2gray(frame) for frame in frames])

        # if transpose:
        #     frames = np.array([frame.T for frame in frames])

        return DataLoader(
            VideosDataset(test_faces, test_emotions),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

    def get_test_set(self) -> dict:
        return self.dataset.pop(random.randint(0, len(self.dataset) - 1))

    def get_train_val_data_loaders(self, gray_scale=False, transpose=True):

        flatten_dataset = self.get_flatten_persons_from_dataset()

        one_offset_emotions = torch.tensor(
            np.array(flatten_dataset["emotions"]) - 1
        ).long()

        one_hot_emotions = nn.functional.one_hot(
            one_offset_emotions, num_classes=config.NUM_EMOTIONS
        ).float()

        faces = self.apply_face_filters(
            flatten_dataset["frames"], gray_scale, transpose
        )

        # frames = np.array(
        #     flatten_dataset["frames"],
        #     dtype=np.float32,
        # )

        # if config.APPLY_HOG:
        #     frames = apply_hog(frames)
        # elif gray_scale:
        #     frames = np.array([rgb2gray(frame) for frame in frames])

        # if transpose:
        #     frames = np.array([frame.T for frame in frames])

        train_faces, val_faces, train_emotions, val_emotions = train_test_split(
            faces, one_hot_emotions, test_size=self.val_set_size
        )

        return (
            DataLoader(
                VideosDataset(train_faces, train_emotions),
                batch_size=self.batch_size,
                shuffle=self.shuffle,
            ),
            DataLoader(
                VideosDataset(val_faces, val_emotions),
                batch_size=self.batch_size,
                shuffle=self.shuffle,
            ),
        )

    def get_frames_and_emotions(self):
        frames, emotions = [], []

        for person in self.dataset:
            for frame, emotion in zip(person["faces"], person["emotions"]):
                frames.append(frame)
                emotions.append(emotion)

        return frames, emotions

    def get_flatten_persons_from_dataset(self):
        frames, emotions = [], []

        for person in self.dataset:
            for frame, emotion in zip(person["faces"], person["emotions"]):
                frames.append(frame)
                emotions.append(emotion)

        return {"frames": frames, "emotions": emotions}

    def get_videos(self, show_frames=False, start_loading_folder=0):
        print("--------Loading videos files--------")

        dataset = []
        for folder in os.listdir(config.VIDEOS_DIRECTORY):
            if config.MAX_FOLDERS is not None and len(dataset) >= config.MAX_FOLDERS:
                break
            # if len(dataset) >= config.MAX_FOLDERS:
            #     break
            # if start_loading_folder < folder[:]:
            #     start_loading_folder -= 1
            #     continue

            print("Loading folder:", folder)

            selected_frames = self.get_one_person_videos(folder)

            frames = np.array([frame["image"] for frame in selected_frames])
            emotions = np.array([frame["emotion"] for frame in selected_frames])

            detected_faces = get_faces_from_images(
                frames,
                show_frames,
                labels_indexes=[frame["emotion"] for frame in selected_frames],
                crop=True,
                only_one_face=True,
            )

            dataset.append({"faces": detected_faces, "emotions": emotions})

        print("--------Videos loaded--------")

        self.dataset = dataset

        return self.dataset

    def get_one_person_videos(self, folder):
        selected_frames = []

        for video_filename in os.listdir(config.VIDEOS_DIRECTORY + folder):
            if (
                config.MAX_VIDEOS_PER_PERSON is not None
                and len(selected_frames) / config.FRAMES_PER_VIDEO
                >= config.MAX_VIDEOS_PER_PERSON
            ):
                break

            emotion_index = int(video_filename[6:8])

            video_path = os.path.join(config.VIDEOS_DIRECTORY + folder, video_filename)

            cap = cv2.VideoCapture(video_path)

            all_frames = []

            while cap.isOpened():
                ret, frame = cap.read()  # frame size: (720, 1280, 3)
                if ret:
                    all_frames.append(frame)
                else:
                    break

            used_frames = []

            fraction_video_size = (len(all_frames) - 1) // config.FRAMES_PER_VIDEO

            offset = 3

            for iteration_offset in range(config.FRAMES_PER_VIDEO):
                i = random.randint(
                    fraction_video_size * iteration_offset + offset,
                    fraction_video_size * (iteration_offset + 1) - offset,
                )

                while i in used_frames:
                    i = random.randint(0, len(all_frames) - 1)

                selected_frames.append(
                    {"image": all_frames[i], "emotion": emotion_index}
                )

                used_frames.append(i)

        return selected_frames

    def show_frame_with_emotion(self, frame, emotion):
        cv2.imshow("frame", frame)
        print("Emotion:", emotion)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_dataset_to_file(self, filename="videos_dataset.npy"):
        np.save(filename, self.dataset)

    def load_dataset_from_file(self, filename="videos_dataset.npy"):
        dataset = np.load(filename, allow_pickle=True)
        self.dataset = dataset.tolist()
        return list(dataset)


if __name__ == "__main__":
    # Loading and saving dataset
    dataset = LoadVideosDataset()

    dataset.get_videos(show_frames=False)

    dataset.print_dataset_info()

    dataset.normalize_dataset()

    dataset.save_dataset_to_file(filename="test2.npy")
