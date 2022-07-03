import numpy
from detector.models.yolo import Yolo
from detector.models.pretrained_pytorch.model import PretrainedModel


class ModelManager:
    """
    A class for all models management.

    Parameters:
        model_name (str): The name of the model that will perform detection on given image
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_detection_successful = False

    def perform_detection_on(self, numpy_image: numpy.ndarray) -> tuple[str, int]:
        """
        Chooses the right one model and performs detecton on given image

        Parameters:
            numpy_image (numpy.ndarray): An image in numpy array format
        """
        if self.model_name in Yolo.AVAILABLE_MODELS:
            yolo = Yolo()
            return self._perform_detection(yolo, numpy_image)
        elif self.model_name in PretrainedModel.AVAILABLE_MODELS.keys():
            pretrained_model = PretrainedModel()
            return self._perform_detection(pretrained_model, numpy_image)
        else:
            raise PretrainedModelNotFoundError(f"Given model name: {self.model_name} has not been found.")

    def _perform_detection(self, initialized_model: any, numpy_image: numpy.ndarray) -> tuple[str, int]:
        initialized_model.load(self.model_name)
        rendered_image, number_of_detections = initialized_model.perform_detection_on(numpy_image)
        self.is_detection_successful = initialized_model.is_detection_successfully_performed
        return rendered_image, number_of_detections


class PretrainedModelNotFoundError(Exception):
    pass