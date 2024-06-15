import os
import random
import numpy as np
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from skimage.color import rgb2gray
from tqdm import tqdm

import config
from landmarks_utility import EyeMouthImages
from image_processing import ImageProcessing
from face_detection_from_photos import get_faces_from_images


class VideosDataset(Dataset):
    def __init__(self, faces, emotions):
        self.faces = torch.tensor(faces)
        self.emotions = torch.tensor(emotions)
        self.n_faces = len(self.faces)

    def __len__(self):
        return self.n_faces

    def __getitem__(self, index):
        return self.faces[index], self.emotions[index]


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

    def print_dataset_info(self, count_emotions=True):
        self.different_emotions_index = set(
            np.array(
                [self.dataset[i]["emotions"] for i in range(len(self.dataset))]
            ).flatten()
        )
        self.different_emotions_names = [
            config.EMOTIONS_DICT[i] for i in self.different_emotions_index
        ]

        print("--------Dataset info--------")
        print("People in dataset:", len(self.dataset))
        print("Frames per person:", len(self.dataset[0]["faces"]))
        print(
            "Frames per emotion from one person:",
            int(len(self.dataset[0]["faces"]) / len(self.different_emotions_index)),
        )
        print("Total frames:", len(self.dataset) * len(self.dataset[0]["faces"]))
        print("Frame shape:", np.array(self.dataset[0]["faces"][0]).shape)
        print("Emotions in dataset:", self.different_emotions_names)
        if count_emotions:
            self.count_emotions()
            print("Emotions counter:", self.emotions_counter)

        print("______________________________")

    def count_emotions(self):
        emotions_counter_index = {
            emotion: 0 for emotion in self.different_emotions_index
        }
        for person in self.dataset:
            for emotion in person["emotions"]:
                emotions_counter_index[emotion] += 1

        self.emotions_counter = {
            config.EMOTIONS_DICT[key]: value
            for key, value in emotions_counter_index.items()
        }

    def normalize_dataset(self):
        for person in self.dataset:
            for i in range(len(person["faces"])):
                person["faces"][i] = person["faces"][i] / 255

    def unnormalize_faces(self, faces):
        for face in faces:
            face *= 255
        return faces.astype("uint8")

    def normalize_faces(self, faces):
        faces = faces.astype("float32")
        for face in faces:
            face /= 255.0
        return faces.astype("float32")

    def apply_face_filters(self, faces=None, gray_scale=False, transpose=True):

        faces = np.array(
            faces,
            dtype=np.float32,
        )

        if config.APPLY_HOG:
            faces = ImageProcessing.apply_hog(faces)
        elif gray_scale or config.GENERATE_MOUTH_AND_EYE:
            faces = np.array([rgb2gray(frame) for frame in faces])

        if transpose:  # ??????????????????????????????????????
            faces = np.array([face.T for face in faces])

        if config.GENERATE_MOUTH_AND_EYE:
            faces = self.unnormalize_faces(faces)
            face_eye_mouth_imgs = []

            for face in faces:
                face = face.astype("uint8")

                eye, mouth = EyeMouthImages.get_eye_mouth_images(face)

                square_eye = ImageProcessing.resize_image_to_square(eye, two_dim=True)
                square_mouth = ImageProcessing.resize_image_to_square(
                    mouth, two_dim=True
                )

                face_eye_mouth_img = np.stack((face, square_eye, square_mouth))
                face_eye_mouth_imgs.append(face_eye_mouth_img)

            faces = np.array(face_eye_mouth_imgs)

            faces = self.normalize_faces(faces)

        return np.array(faces)

    def get_test_data_loader(self, gray_scale=False, transpose=True):

        dataset_len = len(self.dataset) * len(self.dataset[0]["faces"])

        test_set_len = 0
        test_frames = {"faces": [], "emotions": []}

        while test_set_len < dataset_len * 0.1:
            if len(self.dataset) == 0:
                raise Exception("No more frames to load to test set - dataset is empty")

            frames = self.get_test_set()

            if len(test_frames["faces"]) == 0:
                test_frames["faces"] = frames["faces"]
            else:
                test_frames["faces"] = np.concatenate(
                    (test_frames["faces"], frames["faces"]), axis=0
                )

            test_frames["emotions"] = np.append(
                test_frames["emotions"], frames["emotions"]
            )

            test_set_len = len(test_frames["faces"])

        test_emotions = test_frames["emotions"]

        test_faces = self.apply_face_filters(
            test_frames["faces"], gray_scale, transpose
        )

        return DataLoader(
            VideosDataset(test_faces, test_emotions),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

    def get_test_set(self) -> dict:
        return self.dataset.pop(random.randint(0, len(self.dataset) - 1))

    def get_train_val_data_loaders(self, gray_scale=False, transpose=True):

        flatten_dataset = self.get_flatten_persons_from_dataset()

        adjusted_emotions = torch.tensor(
            ImageProcessing.adjust_indexes_emotions(flatten_dataset["emotions"])
        ).long()

        one_hot_emotions = nn.functional.one_hot(
            adjusted_emotions, num_classes=config.NUM_EMOTIONS
        ).float()

        faces = self.apply_face_filters(
            flatten_dataset["frames"], gray_scale, transpose
        )

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

    def generate_face_eye_mouth_images(self, apply_hog=False, show_progress=False):
        print("--------Generating face, eye and mouth images--------")

        for person_i, person in enumerate(tqdm(self.dataset, desc="Persons")):
            if show_progress:
                tqdm.write(f"---Processing person {person_i + 1}/{len(self.dataset)}")

            one_person_faces = person["faces"]

            one_person_faces = np.array(
                one_person_faces,
                dtype=np.float32,
            )

            if apply_hog:
                one_person_faces = ImageProcessing.apply_hog(one_person_faces)
            else:
                one_person_faces = np.array(
                    [rgb2gray(frame) for frame in one_person_faces]
                )

            one_person_faces = self.unnormalize_faces(one_person_faces)

            temp_face_eye_mouth_imgs = []

            for face_i, face in enumerate(
                tqdm(one_person_faces, leave=False, desc="Faces")
            ):
                if show_progress:
                    tqdm.write(f"Processing face {face_i + 1}/{len(one_person_faces)}")

                face = face.astype("uint8")

                eye_img, mouth_img = EyeMouthImages.get_eye_mouth_images(face)

                square_eye_img = ImageProcessing.resize_image_to_square(
                    eye_img, two_dim=True
                )
                square_mouth_img = ImageProcessing.resize_image_to_square(
                    mouth_img, two_dim=True
                )

                face_eye_mouth_img = np.stack((face, square_eye_img, square_mouth_img))
                temp_face_eye_mouth_imgs.append(face_eye_mouth_img)

            one_person_face_eye_mouth_imgs = np.array(temp_face_eye_mouth_imgs)

            one_person_face_eye_mouth_imgs = self.normalize_faces(
                one_person_face_eye_mouth_imgs
            )

            person["faces"] = one_person_face_eye_mouth_imgs

    def get_videos(self, show_frames=False):
        print("--------Loading videos files--------")

        dataset = []
        for folder in os.listdir(config.VIDEOS_DIRECTORY):
            if config.MAX_FOLDERS is not None and len(dataset) >= config.MAX_FOLDERS:
                break

            print(
                f"---Loading folder: {folder} "
                f"[{folder[-2:]} / {min(len(os.listdir(config.VIDEOS_DIRECTORY)), config.MAX_FOLDERS)}]"
            )

            selected_frames = self.get_one_person_videos(folder)

            frames = np.array([frame["image"] for frame in selected_frames])
            emotions = np.array([frame["emotion"] for frame in selected_frames])

            detected_faces = get_faces_from_images(
                frames,
                show_frames,
                labels_indexes=emotions,
                crop=True,
                only_one_face=True,
                folder=folder,
            )

            if len(detected_faces) == 0:
                continue

            dataset.append({"faces": detected_faces, "emotions": emotions})

        print("--------Videos loaded--------")
        print("_____________________________")

        self.dataset = dataset

        return self.dataset

    def get_one_person_videos(self, folder):
        selected_frames = []

        emotions_counter = {1: 0, 3: 0, 4: 0, 5: 0}

        for video_filename in os.listdir(config.VIDEOS_DIRECTORY + folder):
            if (
                config.MAX_VIDEOS_PER_PERSON is not None
                and len(selected_frames) / config.FRAMES_PER_VIDEO
                >= config.MAX_VIDEOS_PER_PERSON
            ):
                break

            emotion_index = int(video_filename[6:8])

            if emotion_index not in config.EMOTIONS_DICT:
                continue

            if (
                emotion_index != 1
                and emotions_counter[1] <= emotions_counter[emotion_index]
            ):
                continue

            emotions_counter[emotion_index] += 1

            video_path = os.path.join(config.VIDEOS_DIRECTORY + folder, video_filename)

            cap = cv2.VideoCapture(video_path)

            all_frames = []

            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    all_frames.append(frame)
                else:
                    break

            used_frames = []

            fraction_video_size = (len(all_frames) - 1) // config.FRAMES_PER_VIDEO

            overlap_offset = 3

            for iteration_offset in range(config.FRAMES_PER_VIDEO):
                i = random.randint(
                    fraction_video_size * iteration_offset + overlap_offset,
                    fraction_video_size * (iteration_offset + 1) - overlap_offset,
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
        path = os.path.join(config.DATASETS_DIRECTORY, filename)
        np.save(path, self.dataset)
        print("--------Dataset saved--------")

    def load_dataset_from_file(self, filename="videos_dataset.npy"):
        path = os.path.join(config.DATASETS_DIRECTORY, filename)
        dataset = np.load(path, allow_pickle=True)
        print("--------Dataset loaded--------")
        self.dataset = dataset.tolist()
        return list(dataset)


if __name__ == "__main__":

    # Loading videos and saving dataset

    dataset = LoadVideosDataset()

    # dataset.get_videos(show_frames=False)

    # dataset.print_dataset_info()

    # dataset.normalize_dataset()

    dataset.load_dataset_from_file("full_dataset_size_100_frames_8_equal_classes.npy")

    dataset.print_dataset_info()

    dataset.generate_face_eye_mouth_images()

    dataset.print_dataset_info()

    dataset.save_dataset_to_file(filename="test.npy")
