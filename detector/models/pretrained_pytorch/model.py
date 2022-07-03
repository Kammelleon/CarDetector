import base64
import traceback
from typing import Union

import numpy
from torchvision.models import detection
import numpy as np
import pickle
import torch
import cv2


class PretrainedModel:
    """
    Base class for all PyTorch pretrained models.

    Attributes:
        coco_dataset_location (str): path (relative or absolute) to COCO dataset file in pickle format
    """
    def __init__(self, coco_dataset_location: str = "./detector/models/pretrained_pytorch/coco_dataset.pickle"):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CLASSES = None
        self.COLORS = None
        self.AVAILABLE_MODELS = {
            "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
            "high-res-frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_fpn,
            "low-res-frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
            "retinanet": detection.retinanet_resnet50_fpn,
            "fcos-resnet": detection.fcos_resnet50_fpn,
            "ssd300": detection.ssd300_vgg16,
            "ssdlite-mobilenet": detection.ssdlite320_mobilenet_v3_large
        }
        self.minimum_confidence = 0.6
        self.coco_dataset_location = coco_dataset_location
        self.model = None

        self.is_model_loaded = False
        self.is_detection_successfully_performed = False

        self._load_dataset(coco_dataset_location)
        self._generate_colors_for_bounding_boxes()

    def load(self, pretrained_model_name: str) -> None:
        """
        Loads given model into memory.

        Parameters:
            pretrained_model_name (str): model name from the AVAILABLE_MODELS keys
        """
        try:
            self.model = self.AVAILABLE_MODELS[pretrained_model_name](pretrained=True,
                                                                      progress=True,
                                                                      num_classes=len(self.CLASSES),
                                                                      pretrained_backbone=True).to(self.DEVICE)
            self.model.eval()
            self.is_model_loaded = True
        except KeyError:
            self.is_model_loaded = False
            raise PretrainedModelNotFoundError(f"Selected model: {pretrained_model_name} has not been found in "
                                               f"PyTorch pretrained models library")
        except Exception:
            self.is_model_loaded = False
            raise Exception(f"Another exception occured: {traceback.format_exc()}")

    def perform_detection_on(self, image: numpy.ndarray) -> tuple[str, int]:
        """
        Preprocesses given image and performs detection on that image.

        Parameters:
            image (numpy.ndarray): image converted into numpy array.

        Returns:
            tuple (str, int): Tuple - Image in base64 format as string and number of detections
        """
        preprocessed_image, original_image = self._preprocess_image(image)

        detections = self.model(preprocessed_image)[0]  # Perform detection

        number_of_detections = 0

        for i in range(0, len(detections["boxes"])):

            confidence = detections["scores"][i]

            if confidence > self.minimum_confidence:

                idx = int(detections["labels"][i])

                # Shift index by -1 because of labeling problems (?)
                idx = idx - 1

                if self.CLASSES[idx] != "car":
                    continue

                number_of_detections += 1

                bounding_box_coordinates = detections["boxes"][i].detach().cpu().numpy()
                (start_x, start_y, end_x, end_y) = bounding_box_coordinates.astype("int")

                # Draw bounding box on original image
                cv2.rectangle(original_image, (start_x, start_y), (end_x, end_y),
                              self.COLORS[idx], 2)

                # Put label for bounding box
                label = f"{self.CLASSES[idx]}: {confidence * 100:.2f}%"
                y = start_y - 15 if start_y - 15 > 15 else start_y + 15

                cv2.putText(original_image, label, (start_x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)

        # Base64 format of image is required to show detection in HTML <img> tag
        base64_image = self._ndarray_to_base64_string(original_image)

        self.is_detection_successfully_performed = True
        return base64_image, number_of_detections

    def _load_dataset(self, coco_dataset_location: str) -> None:
        """
        Loads COCO dataset from given location.

        Parameters:
            coco_dataset_location (str): path (relative or absolute) to COCO dataset file in pickle format
        """
        try:
            self.CLASSES = pickle.loads(open(coco_dataset_location, "rb").read())
            if len(self.CLASSES) != 91:
                raise DatasetError(f"Dataset must contain exactly 91 classes. Your dataset contains: {len(self.CLASSES)}")
        except FileNotFoundError:
            raise DatasetNotFoundError(f"Dataset has not been found in given location: {self.coco_dataset_location}")
        except pickle.UnpicklingError:
            raise DatasetError(f"Cannot unpickle given dataset: {self.coco_dataset_location}")

    def _generate_colors_for_bounding_boxes(self) -> None:
        """
        Generates random colors for bounding boxes for every class in COCO dataset
        """
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

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

    def _preprocess_image(self, numpy_image: Union[numpy.ndarray, str], image_from_numpy: bool = True) -> tuple[torch.Tensor, numpy.ndarray]:
        """
        Converts numpy ndarray into image-object and prepares image for detection.

        Parameters:
            numpy_image (numpy.ndarray): An image in numpy array format
            image_from_numpy (bool): (Optional) Set to False if you want to read from raw image (not from numpy array)

        Returns:
            tuple (torch.Tensor, numpy.ndarray): Tuple - Image in tensor format and original image in numpy array format
        """
        try:
            if image_from_numpy:
                image = cv2.imdecode(numpy_image, cv2.IMREAD_UNCHANGED)
            else:
                image = cv2.imread(numpy_image, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except cv2.error:
            self.is_detection_successfully_performed = False
            raise ImageLoadError("Ensure your image exists in given location or is correct image-like file.")
        try:
            original_image = image.copy()

            image = image.transpose((2, 0, 1))

            image = np.expand_dims(image, axis=0)
            image = image / 255.0
            image = torch.FloatTensor(image)

            preprocessed_image = image.to(self.DEVICE)

            return preprocessed_image, original_image
        except Exception:
            self.is_detection_successfully_performed = False
            print("Another exception:")
            raise Exception("Error during processing image")


class PretrainedModelNotFoundError(Exception):
    pass

class DatasetNotFoundError(Exception):
    pass

class DatasetError(Exception):
    pass

class ImageLoadError(Exception):
    pass

class ImageConversionError(Exception):
    pass