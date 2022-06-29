import cv2
import numpy as np
import pytest
from detector.yolo import Yolo


class TestClass:

    @pytest.fixture
    def yolo_model(self):
        return Yolo()

    @pytest.fixture
    def yolo_loaded_model(self, yolo_model):
        yolo = yolo_model
        yolo.load("yolov5n")
        return yolo

    def test_load_correct_model_type(self, yolo_model):
        # given
        yolo = yolo_model

        # when
        yolo.load("yolov5n")

        # then
        assert yolo.is_model_loaded is True

    def test_load_incorrect_model_type(self, yolo_model):
        yolo = yolo_model

        yolo.load("incorrect_yolo_model")

        assert yolo.is_model_loaded is False

    def test_detection_performed_on_valid_image(self, yolo_loaded_model):
        # given
        image = "../test_files/car.jpg"

        # when
        yolo_loaded_model.perform_detection_on(image, image_from_np_string=False)

        # then
        assert yolo_loaded_model.is_detection_successfully_performed is True

    def test_detection_performed_on_invalid_image(self, yolo_loaded_model):
        # given
        image = "../test_files/car_empty.jpg"

        # when
        yolo_loaded_model.perform_detection_on(image, image_from_np_string=False)

        # then
        assert yolo_loaded_model.is_detection_successfully_performed is False
