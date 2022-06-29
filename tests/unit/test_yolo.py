import pytest
from detector.yolo import Yolo


@pytest.fixture
def yolo_model():
    return Yolo()


def test_load_correct_model_type(yolo_model):
    # given
    yolo = yolo_model

    # when
    yolo.load("yolov5s")

    # then
    assert yolo.is_model_loaded is True


def test_raise_exception_if_incorrect_model_type(yolo_model):
    yolo = yolo_model

    yolo.load("incorrect_yolo_model")

    assert yolo.is_model_loaded is False
