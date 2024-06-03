import os
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from tqdm import tqdm
import torch
from torch import nn
from sklearn.metrics import f1_score, confusion_matrix

import config
from videos_dataset import LoadVideosDataset
from cnn_models import Model3D200, Model2D100


class EvaluateModel:
    def __init__(self, model=None, epochs=40, lr=0.01):
        self.epochs = epochs
        self.lr = lr

        self.model = model
        self.criterion = None
        self.optimizer = None

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

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def to_cuda(self):
        if self.model is not None and torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def load_data(self, data_filename) -> None:

        from_file = False if data_filename is None else True

        dataset = LoadVideosDataset()

        if from_file:
            dataset.load_dataset_from_file(data_filename)
        else:
            dataset.get_videos(show_frames=False)
            dataset.normalize_dataset()
            dataset.save_dataset_to_file()

        dataset.apply_filters()

        dataset.print_dataset_info()

        self.test_loader = dataset.get_test_data_loader(
            gray_scale=self.gray_scale, transpose=self.transpose
        )

        self.train_loader, self.val_loader = dataset.get_train_val_data_loaders(
            gray_scale=self.gray_scale, transpose=self.transpose
        )

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
            best_accuracy = 0.0
            best_model = None
            for epoch in tqdm(range(self.epochs), desc="Training"):
                # Initialize metrics
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                val_loss = 0.0
                val_correct = 0
                val_total = 0

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

                # Validation phase
                self.model.eval()
                with torch.no_grad():
                    for x_val, y_val in self.val_loader:

                        if self.unsqueeze:
                            x_val = x_val.unsqueeze(1)

                        x_val = x_val.cuda()
                        y_val = y_val.cuda()

                        val_outputs = self.model(x_val)

                        # for i in range(10):
                        #     print("val_outputs:", val_outputs[i])
                        #     print("y_val:", y_val[i])
                        #     print("-----------")

                        val_loss += self.criterion(val_outputs, y_val).item()
                        val_correct += self.count_similar(val_outputs, y_val)
                        val_total += len(y_val)

                val_accuracy = val_correct / val_total

                if best_accuracy < val_accuracy:
                    best_accuracy = val_accuracy
                    best_model = self.model.state_dict()

                if (epoch + 1) % 1 == 0:
                    tqdm.write(
                        f"Epoch [{epoch+1}/{self.epochs}], "
                        f"Train Loss: {train_loss/len(self.train_loader):.4f}, "
                        f"Train Acc: {train_accuracy:.4f}, "
                        f"Val Loss: {val_loss/len(self.val_loader):.4f}, "
                        f"Val Acc: {val_accuracy:.4f}"
                    )

            self.model.load_state_dict(best_model)

    def predict(self, X):
        pred = self.model(X.cuda())
        _, pred = torch.max(pred, 1)
        pred = pred + 1
        return pred

    def get_emotion_from_tensor(self, tensor):
        return config.EMOTIONS_DICT[tensor.item()]

    def test_model(self, show_plots=True) -> None:

        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        all_preds = []
        all_labels = []

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for x_test, y_test in self.test_loader:
                x_test = x_test.cuda()
                y_test = y_test.cuda()

                if self.unsqueeze:
                    x_test = x_test.unsqueeze(1)

                test_outputs = self.predict(x_test)

                test_outputs = test_outputs.float()
                y_test = y_test.float()

                loss = criterion(test_outputs, y_test)

                test_loss += loss.item()

                test_correct += torch.sum(test_outputs == y_test).item()
                test_total += len(y_test)

                all_preds.extend(test_outputs.cpu().numpy())
                all_labels.extend(y_test.cpu().numpy())

        test_accuracy = test_correct / test_total
        f1 = f1_score(all_labels, all_preds, average="weighted")

        print(
            f"Test Loss: {test_loss/len(self.test_loader):.4f}, Test Acc: {test_accuracy:.4f}, F1 Score: {f1:.4f}"
        )
        if show_plots:
            self.plot_confusion_matrix(all_labels, all_preds)

    def plot_confusion_matrix(self, labels, preds):

        print("different labels:", set(labels))
        print("different preds:", set(preds))

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
        plt.show()
        plt.savefig("confusion_matrix.png")

    def save_model_to_file(self, filename="emotions_recognition_model.pt") -> None:
        file_path = os.path.join(config.MODELS_DIRECTORY, filename)
        torch.save(self.model.state_dict(), file_path)
        print("--------Model saved--------")

    def load_model(self, filename="emotions_recognition_model.pt"):
        try:
            file_path = os.path.join(config.MODELS_DIRECTORY, filename)
            self.model.load_state_dict(torch.load(file_path))
            self.init_model_parameters()
        except FileNotFoundError:
            print(f'File "{filename}" with model not found')
        except:
            print("An exception occurred while loading model from file")
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

    ev1 = EvaluateModel(model=Model3D200())
    ev1.load_train_save_model(
        data_filename="test.npy",
        save_model_filename="test_3D.pt",
        show_test_plots=True,
    )

    # ev2 = EvaluateModel(model=Model2D100())
    # ev2.load_train_save_model(
    #     data_filename="test.npy",
    #     save_model_filename="test_2D.pt",
    #     show_test_plots=True,
    # )
