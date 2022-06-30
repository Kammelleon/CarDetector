class ModelManager:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def perform_detection_on(self, numpy_image):
        if "yolov5" in self.model_name:
            from detector.models.yolo import Yolo
            yolo = Yolo()
            yolo.load(self.model_name)
            successfully_performed_detection, rendered_image = yolo.perform_detection_on(numpy_image)
            if successfully_performed_detection:
                return successfully_performed_detection, rendered_image
            else:
                return False, None
        else:
            from detector.models.pretrained_torch.model import PretrainedModel
            pretrained_model = PretrainedModel()
            pretrained_model.load(self.model_name)
            successfully_performed_detection, rendered_image = pretrained_model.perform_detection_on(numpy_image)
            if successfully_performed_detection:
                return successfully_performed_detection, rendered_image
            else:
                return False, None

