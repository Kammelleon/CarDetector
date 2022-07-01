import base64
import traceback

import numpy
from torchvision.models import detection
import numpy as np
import pickle
import torch
import cv2


class PretrainedModel:
    def __init__(self, coco_dataset_location="./detector/models/pretrained_pytorch/coco_dataset.pickle"):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CLASSES = None
        self.COLORS = None
        self.MINIMUM_CONFIDENCE = 0.7
        self.coco_dataset_location = coco_dataset_location
        self.MODELS = {
            "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
            "high-res-frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_fpn,
            "low-res-frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
            "retinanet": detection.retinanet_resnet50_fpn,
            "fcos-resnet": detection.fcos_resnet50_fpn,
            "ssd300": detection.ssd300_vgg16,
            "ssdlite-mobilenet": detection.ssdlite320_mobilenet_v3_large
        }

        self.model = None
        self.is_model_loaded = False
        self.is_detection_successfully_performed = False

        self._load_dataset(coco_dataset_location)
        self._generate_colors()

    def load(self, pretrained_model_name: str) -> None:
        try:
            self.model = self.MODELS[pretrained_model_name](pretrained=True,
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

    def perform_detection_on(self, image: numpy.ndarray) -> str:
        preprocessed_image, original_image = self._preprocess_image(image)

        detections = self.model(preprocessed_image)[0]

        for i in range(0, len(detections["boxes"])):

            confidence = detections["scores"][i]

            if confidence > self.MINIMUM_CONFIDENCE:

                idx = int(detections["labels"][i])

                # Shift index by -1 because of labeling problems (?)
                idx = idx - 1

                if self.CLASSES[idx] != "car":
                    continue

                bounding_box_coordinates = detections["boxes"][i].detach().cpu().numpy()
                (start_x, start_y, end_x, end_y) = bounding_box_coordinates.astype("int")

                label = f"{self.CLASSES[idx]}: {confidence * 100:.2f}%"
                print("[INFO] {}".format(label))

                # Draw bounding box on original image
                cv2.rectangle(original_image, (start_x, start_y), (end_x, end_y),
                              self.COLORS[idx], 2)

                # Put label for bounding box
                y = start_y - 15 if start_y - 15 > 15 else start_y + 15
                cv2.putText(original_image, label, (start_x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)

        # Base64 format of image is required to show detection in HTML <img> tag
        base64_image = self._ndarray_to_base64_string(original_image)

        self.is_detection_successfully_performed = True

        return self.is_detection_successfully_performed, base64_image

    def _load_dataset(self, coco_dataset_location: str) -> None:
        try:
            self.CLASSES = pickle.loads(open(coco_dataset_location, "rb").read())
            if len(self.CLASSES) != 91:
                raise DatasetError(f"Dataset must contain exactly 91 classes. Your dataset contains: {len(self.CLASSES)}")
        except FileNotFoundError:
            raise DatasetNotFoundError(f"Dataset has not been found in given location: {self.coco_dataset_location}")
        except pickle.UnpicklingError:
            raise DatasetError(f"Cannot unpickle given dataset: {self.coco_dataset_location}")

    def _generate_colors(self) -> None:
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

    def _ndarray_to_base64_string(self, ndarray: numpy.ndarray) -> str:
        try:
            img = cv2.cvtColor(ndarray, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.png', img)
            return base64.b64encode(buffer).decode('utf-8')
        except cv2.error:
            raise ImageConversionError("Ensure that your image is correct and is of type numpy.ndarray")

    def _preprocess_image(self, numpy_image: numpy.ndarray or str, image_from_numpy: bool = True) -> tuple[torch.Tensor, numpy.ndarray]:
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
            print("Another exception:")
            print(traceback.print_exc())
            self.is_detection_successfully_performed = False
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