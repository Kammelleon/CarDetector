import base64
import traceback

import numpy
import torch
import cv2


class Yolo:
    def __init__(self):
        self.confidence = 0.7
        self.model = None
        self.is_model_loaded = False
        self.is_detection_successfully_performed = False

    def load(self, model_type: str) -> bool:
        try:
            self.model = torch.hub.load('ultralytics/yolov5', model_type)
            self.model.classes = [2]  # Select only car (index: 2) from COCO dataset while performing detection
            self.is_model_loaded = True
            return self.is_model_loaded
        except RuntimeError:
            self.is_model_loaded = False
            raise YoloModelLoadError(f"Could not load given model type: {model_type}")

    def perform_detection_on(self, image: numpy.ndarray or str, image_from_numpy: bool = True) -> tuple[str, int]:
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

            result = self.model(image)
            number_of_detections = len(result.pandas().xyxy[0])
            print(number_of_detections)
            result.render()
            rendered_image = result.imgs[0]
            base64_image = self._ndarray_to_base64_string(rendered_image)
            self.is_detection_successfully_performed = True
            return base64_image, number_of_detections
        except Exception:
            self.is_detection_successfully_performed = False
            raise Exception(f"Another exception occured: {traceback.format_exc()}")

    def _ndarray_to_base64_string(self,ndarray: numpy.ndarray) -> str:
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