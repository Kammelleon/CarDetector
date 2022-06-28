import torch
import cv2

class Yolo:
    def __init__(self):
        self.confidence = 0.7
        self.model = None

    def load(self, model_type):
        self.model = torch.hub.load('ultralytics/yolov5', model_type)
        self.model.classes = [2]

    def perform_detection_on(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.model(image)
        result.render()
        # result.render()
        # img = cv2.cvtColor(result.imgs[0], cv2.COLOR_RGB2BGR)
        # cv2.imshow("im",img)
        # cv2.waitKey(0)
