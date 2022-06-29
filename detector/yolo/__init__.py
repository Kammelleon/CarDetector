import traceback

import torch
import cv2
from django.shortcuts import redirect
from django.contrib import messages


class Yolo:
    def __init__(self):
        self.confidence = 0.7
        self.model = None
        self.is_model_loaded = False
        self.is_detection_successfully_performed = False

    def load(self, model_type):
        try:
            self.model = torch.hub.load('ultralytics/yolov5', model_type)
            self.model.classes = [2]
            self.is_model_loaded = True
            return self.is_model_loaded
        except RuntimeError:
            self.is_model_loaded = False
            return self.is_model_loaded

    def perform_detection_on(self, image):
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = self.model(image)
            result.render()
            self.is_detection_successfully_performed = True
            return self.is_detection_successfully_performed
        except:
            print(traceback.print_exc())
            self.is_detection_successfully_performed = False
            return self.is_detection_successfully_performed
        # result.render()
        # img = cv2.cvtColor(result.imgs[0], cv2.COLOR_RGB2BGR)
        # cv2.imshow("im",img)
        # cv2.waitKey(0)
