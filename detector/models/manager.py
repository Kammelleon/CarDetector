class ModelManager:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def perform_detection_on(self, numpy_image):
        if "yolov5" in self.model_name:
            from detector.models.yolo import Yolo
            yolo = Yolo()
            is_detection_successfully_performed, rendered_image = self._perform_detection(yolo, numpy_image)
            return is_detection_successfully_performed, rendered_image
        else:
            from detector.models.pretrained_torch.model import PretrainedModel
            pretrained_model = PretrainedModel()
            is_detection_successfully_performed, rendered_image = self._perform_detection(pretrained_model, numpy_image)
            return is_detection_successfully_performed, rendered_image

    def _perform_detection(self, initialized_model, numpy_image):
        initialized_model.load(self.model_name)
        is_detection_successfully_performed, rendered_image = initialized_model.perform_detection_on(numpy_image)
        return is_detection_successfully_performed, rendered_image

