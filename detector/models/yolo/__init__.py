import base64
import traceback
from typing import Union

import numpy
import torch
import cv2


class Yolo:
    """
    Base class for all YOLOv5 models.
    """
    AVAILABLE_MODELS = ["yolov5n","yolov5s","yolov5m","yolov5l","yolov5x"]
    def __init__(self):
        self.minimum_confidence = 0.6
        self.model = None
        self.is_model_loaded = False
        self.is_detection_successfully_performed = False

    def load(self, model_type: str) -> bool:
        """
        Loads given model into memory.

        Parameters:
            model_type (str): Type of YOLOv5 model (e.g.: yolov5n, yolov5s)
        """
        try:
            self.model = torch.hub.load('ultralytics/yolov5', model_type)
            self.model.classes = [2]  # Select only car for detections (index: 2) from COCO dataset
            self.is_model_loaded = True
            return self.is_model_loaded
        except RuntimeError:
            self.is_model_loaded = False
            raise YoloModelLoadError(f"Could not load given model type: {model_type}")

    def perform_detection_on(self, image: Union[numpy.ndarray, str], image_from_numpy: bool = True) -> tuple[str, int]:
        """
        Preprocesses given image and performs detection on that image.

        Parameters:
            image (numpy.ndarray): image converted into numpy array.
            image_from_numpy (bool): (Optional) Set to False if you want to read from raw image (not from numpy array)

        Returns:
            tuple (str, int): Tuple - Image in base64 format as string and number of detections
        """
        try:
            if image_from_numpy:
                image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
            else:
                image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except cv2.error:
            self.is_detection_successfully_performed = False
            raise ImageLoadError("Ensure your image exists in given location or is correct image-like file.")
        try:
            detections = self.model(image)  # Perform detection
            number_of_detections = len(detections.pandas().xyxy[0])
            detections.render()  # Draw bounding boxes on top of the image
            rendered_image = detections.imgs[0]
            base64_image = self._ndarray_to_base64_string(rendered_image)
            self.is_detection_successfully_performed = True
            return base64_image, number_of_detections
        except Exception:
            self.is_detection_successfully_performed = False
            raise Exception(f"Another exception occured: {traceback.format_exc()}")

    @staticmethod
    def _ndarray_to_base64_string(ndarray: numpy.ndarray) -> str:
        """
        Converts numpy ndarray into base64 string (required for HTML <img> tag)

        Parameters:
            ndarray (numpy.ndarray): An image in numpy array format

        Returns:
            str: String containing base64 image
        """
        try:
            img = cv2.cvtColor(ndarray, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.png', img)
            return base64.b64encode(buffer).decode('utf-8')
        except cv2.error:
            raise ImageConversionError("Ensure that your image is correct and is of type numpy.ndarray")

class YoloModelLoadError(Exception):
    pass

class ImageLoadError(Exception):
    pass

class ImageConversionError(Exception):
    pass