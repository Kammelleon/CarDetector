class ModelManager:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_detection_successful = False

    def perform_detection_on(self, numpy_image):
        if "yolov5" in self.model_name:
            from detector.models.yolo import Yolo
            yolo = Yolo()
            return self._perform_detection(yolo, numpy_image)
        else:
            from detector.models.pretrained_pytorch.model import PretrainedModel
            pretrained_model = PretrainedModel()
            return self._perform_detection(pretrained_model, numpy_image)

    def _perform_detection(self, initialized_model, numpy_image):
        initialized_model.load(self.model_name)
        rendered_image, number_of_detections = initialized_model.perform_detection_on(numpy_image)
        self.is_detection_successful = initialized_model.is_detection_successfully_performed
        return rendered_image, number_of_detections

