import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from torch import nn
from sklearn.metrics import f1_score, confusion_matrix

import config
from image_processing import ImageProcessing
from videos_dataset import LoadVideosDataset
from cnn_models import Model3D100, Model3D200, Model2D100


class EvaluateModel:
    def __init__(self, model=None, epochs=50, lr=0.01, eye_mouth_images=False):
        self.epochs = epochs
        self.lr = lr

        self.model = model
        self.criterion = None
        self.optimizer = None

        self.eye_mouth_images = eye_mouth_images

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_f1_scores = []

        self.init_model_parameters()

    def init_model_parameters(self):
        if self.model is None:
            return

        self.to_cuda()

        if self.model.__class__ == Model2D100:
            self.gray_scale = True
            self.transpose = False
            self.unsqueeze = True
        else:
            self.gray_scale = False
            self.transpose = True
            self.unsqueeze = False

        if self.eye_mouth_images:
            self.transpose = False

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def to_cuda(self):
        if self.model is not None and torch.cuda.is_available():
            self.model = self.model.to("cuda")

    @staticmethod
    def count_similar(y_pred, y_true, show=False):

        _, y_pred = torch.max(y_pred, 1)
        _, y_true = torch.max(y_true, 1)

        sum = torch.sum(y_pred == y_true).item()

        if show:
            print(f"y_true: {y_true}")
            print(f"y_pred: {y_pred}")
            print(f"sum: {sum}")
            print("-------------------")
        return sum

    def train_model(self) -> None:
        with torch.cuda.device(0):
            self.val_accuracy = 0.0
            best_epoch = 0
            best_model = None
            for epoch in tqdm(range(self.epochs), desc="Training"):
                # Initialize metrics
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                val_loss = 0.0
                val_correct = 0
                val_total = 0

                all_preds = []
                all_labels = []

                # Training phase
                self.model.train()
                for x, y in self.train_loader:
                    self.optimizer.zero_grad()

                    if self.unsqueeze:
                        x = x.unsqueeze(1)

                    x_train = x.cuda()
                    y_train = y.cuda()

                    train_outputs = self.model(x_train)

                    loss = self.criterion(train_outputs, y_train)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()
                    train_correct += self.count_similar(train_outputs, y_train)
                    train_total += len(y_train)

                train_accuracy = train_correct / train_total
                self.train_losses.append(train_loss / len(self.train_loader))
                self.train_accuracies.append(train_accuracy)

                # Validation phase
                self.model.eval()
                with torch.no_grad():
                    for x_val, y_val in self.val_loader:

                        if self.unsqueeze:
                            x_val = x_val.unsqueeze(1)

                        x_val = x_val.cuda()
                        y_val = y_val.cuda()

                        val_outputs = self.model(x_val)

                        val_loss += self.criterion(val_outputs, y_val).item()
                        val_correct += self.count_similar(val_outputs, y_val)
                        val_total += len(y_val)

                        all_preds.extend(val_outputs.cpu().numpy())
                        all_labels.extend(y_val.cpu().numpy())

                current_val_accuracy = val_correct / val_total
                self.val_losses.append(val_loss / len(self.val_loader))
                self.val_accuracies.append(current_val_accuracy)
                self.val_f1_scores.append(
                    f1_score(all_labels, all_preds, average="weighted")
                )

                if self.val_accuracy < current_val_accuracy:
                    self.val_accuracy = current_val_accuracy
                    best_epoch = epoch
                    best_model = self.model.state_dict()

                if (epoch + 1) % 1 == 0:
                    tqdm.write(
                        f"Epoch [{epoch+1}/{self.epochs}], "
                        f"Train Loss: {train_loss/len(self.train_loader):.4f}, "
                        f"Train Acc: {train_accuracy:.4f}, "
                        f"Val Loss: {val_loss/len(self.val_loader):.4f}, "
                        f"Val Acc: {current_val_accuracy:.4f}, "
                        f"Val F1: {self.val_f1_scores[-1]:.4f}"
                    )

            print(
                f"\n-------------------------------------------------\n"
                f"Best model accuracy: {self.val_accuracy}, in epoch: {best_epoch+1}\n"
                f"-------------------------------------------------\n"
            )

            self.model.load_state_dict(best_model)

    def test_model(self, show_plots=True) -> None:

        self.model.eval()
        test_correct = 0
        test_total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_test, y_test in self.test_loader:

                x_test = x_test.cuda()
                y_test = y_test.cuda()

                if self.unsqueeze:
                    x_test = x_test.unsqueeze(1)

                test_outputs = self.predict(x_test)

                test_outputs = torch.tensor(
                    ImageProcessing.readjust_indexes_emotions(test_outputs.cpu())
                ).float()
                y_test = y_test.float().cpu()

                test_correct += torch.sum(test_outputs == y_test).item()
                test_total += len(y_test)

                all_preds.extend(test_outputs.cpu().numpy())
                all_labels.extend(y_test.cpu().numpy())

        self.test_accuracy = test_correct / test_total
        self.test_f1 = f1_score(all_labels, all_preds, average="weighted")

        print(
            f"-------------------------------------------------\n"
            f"Test accuracy: {self.test_accuracy:.4f}, Test F1 Score: {self.test_f1:.4f}\n"
            f"-------------------------------------------------\n"
        )
        if show_plots:
            self.plot_confusion_matrix(all_labels, all_preds)
            self.plot_metrics()

    def predict(self, X):
        pred = self.model(X.cuda())
        _, pred = torch.max(pred, 1)
        return pred

    def get_emotion_from_tensor(self, tensor):
        return config.EMOTIONS_DICT[tensor.item()]

    def plot_confusion_matrix(self, labels, preds):

        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=config.EMOTIONS_DICT.values(),
            yticklabels=config.EMOTIONS_DICT.values(),
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        plt.show()

    def plot_metrics(self):
        epochs_range = range(self.epochs)

        plt.figure(figsize=(14, 5))

        plt.subplot(1, 3, 1)
        plt.plot(epochs_range, self.train_losses, label="Training Loss")
        plt.plot(epochs_range, self.val_losses, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")

        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, self.train_accuracies, label="Training Accuracy")
        plt.plot(epochs_range, self.val_accuracies, label="Validation Accuracy")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Accuracy")

        plt.subplot(1, 3, 3)
        plt.plot(epochs_range, self.val_f1_scores, label="Validation F1 Score")
        plt.legend(loc="upper right")
        plt.title("Validation F1 Score")

        plt.tight_layout()
        plt.savefig("training_validation_metrics.png")
        plt.show()

    def load_data(self, data_filename) -> None:

        from_file = False if data_filename is None else True

        dataset = LoadVideosDataset()

        if from_file:
            dataset.load_dataset_from_file(data_filename)
        else:
            dataset.get_videos(show_frames=False)
            dataset.normalize_dataset()
            dataset.save_dataset_to_file()

        dataset.print_dataset_info()

        self.test_loader = dataset.get_test_data_loader(
            gray_scale=self.gray_scale, transpose=self.transpose
        )

        self.train_loader, self.val_loader = dataset.get_train_val_data_loaders(
            gray_scale=self.gray_scale, transpose=self.transpose
        )

    def save_model_to_file(self, filename="emotions_recognition_model.pt") -> None:
        file_path = os.path.join(config.MODELS_DIRECTORY, filename)
        torch.save(self.model.state_dict(), file_path)
        print("--------Model saved--------")

    def load_model(self, filename="emotions_recognition_model.pt", model=Model3D100()):
        try:

            file_path = os.path.join(config.MODELS_DIRECTORY, filename)
            self.model = model
            self.model.load_state_dict(torch.load(file_path))
            self.init_model_parameters()
        except FileNotFoundError:
            raise FileNotFoundError(f'File "{filename}" with model not found')
        except:
            raise Exception("An exception occurred while loading model from file")
        print("--------Model loaded--------")

    def load_train_save_model(
        self, data_filename=None, save_model_filename=None, show_test_plots=False
    ) -> None:
        self.load_data(data_filename)
        print("--------Starting training new model--------")
        self.train_model()
        self.save_model_to_file(save_model_filename)
        self.test_model(show_test_plots)


if __name__ == "__main__":

    # Evalutating and saving models
    ev1 = EvaluateModel(
        model=Model3D100(), epochs=150, lr=0.005, eye_mouth_images=True
    )  # dlaczego tutaj nie chcę robić transpozycji?????????????????
    ev1.load_train_save_model(
        data_filename="full_FEM_size100_frames8.npy",
        save_model_filename="full_dataset_size100_frames8_epochs150.pt",
        show_test_plots=True,
    )
