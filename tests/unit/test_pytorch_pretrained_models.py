import pytest
from detector.models.pretrained_pytorch.model import PretrainedModel, PretrainedModelNotFoundError, DatasetError, \
    DatasetNotFoundError


class TestClass:
    @pytest.fixture
    def pretrained_model(self):
        return PretrainedModel(coco_dataset_location="../../detector/models/pretrained_pytorch/coco_dataset.pickle")

    def test_load_correct_model_type(self, pretrained_model):
        # given
        pytorch_pretrained_model = pretrained_model

        # when
        pytorch_pretrained_model.load("ssd300")

        # then
        assert pytorch_pretrained_model.model is not None
        assert pytorch_pretrained_model.is_model_loaded is True


    def test_load_incorrect_model_type(self, pretrained_model):
        # given
        pytorch_pretrained_model = pretrained_model

        # when
        with pytest.raises(PretrainedModelNotFoundError):
            pytorch_pretrained_model.load("incorrect_pytorch_model")

        # then
        assert pytorch_pretrained_model.model is None
        assert pytorch_pretrained_model.is_model_loaded is False

    def test_load_dataset_that_doesnt_exists_in_given_location(self):
        with pytest.raises(DatasetNotFoundError):
            PretrainedModel(coco_dataset_location="bad_dataset_location")

    def test_load_damaged_dataset(self):
        with pytest.raises(DatasetError):
            PretrainedModel(coco_dataset_location="../test_files/wrong_coco_dataset.pickle")

    def test_load_dataset_that_doesnt_contain_91_classes(self):
        with pytest.raises(DatasetError):
            PretrainedModel(coco_dataset_location="../test_files/90_class_coco_dataset.pickle")

