import cv2
import numpy as np
import pytest
from detector.models.manager import ModelManager, PretrainedModelNotFoundError


class TestClass:
    def test_raise_error_on_nonexistent_model_name(self):
        model_manager = ModelManager("model_that_doesnt_exists")
        numpy_arr = np.array([2,3,1,0])
        with pytest.raises(PretrainedModelNotFoundError):
            model_manager.perform_detection_on(numpy_arr)