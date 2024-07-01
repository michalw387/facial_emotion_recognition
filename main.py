from model_evaluation import EvaluateModel
from video_capture import video_capture

from cnn_models import Model3D100

if __name__ == "__main__":
    ev1 = EvaluateModel()

    ev1.load_model(
        "full_size100_frames8_epochs150_withoutEM_acc72.pt", model=Model3D100()
    )

    video_capture(ev1)
