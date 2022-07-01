import pytest
from detector.models.yolo import Yolo, YoloModelLoadError, ImageConversionError, ImageLoadError


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
        assert yolo.model is not None
        assert yolo.is_model_loaded is True

    def test_load_incorrect_model_type(self, yolo_model):
        # given
        yolo = yolo_model

        # when
        with pytest.raises(YoloModelLoadError):
            yolo.load("incorrect_yolo_model")

        # then
        assert yolo.is_model_loaded is False

    def test_detection_performed_on_valid_image(self, yolo_loaded_model):
        # given
        image = "../test_files/car.jpg"

        # when
        yolo_loaded_model.perform_detection_on(image, image_from_numpy=False)

        # then
        assert yolo_loaded_model.is_detection_successfully_performed is True

    def test_detection_performed_on_invalid_image(self, yolo_loaded_model):
        # given
        image = "../test_files/car_empty.jpg"

        # when
        with pytest.raises(ImageLoadError):
            yolo_loaded_model.perform_detection_on(image, image_from_numpy=False)

        # then
        assert yolo_loaded_model.is_detection_successfully_performed is False

    def test_numpy_to_base64_conversion_on_wrong_image(self, yolo_loaded_model):
        image = "invalid_image"
        with pytest.raises(ImageConversionError):
            yolo_loaded_model._ndarray_to_base64_string(image)

