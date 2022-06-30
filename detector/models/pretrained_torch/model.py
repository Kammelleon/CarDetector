import base64

from torchvision.models import detection
import numpy as np
import pickle
import torch
import cv2


class PretrainedModel:
    def __init__(self):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CLASSES = pickle.loads(open("./detector/models/pretrained_torch/coco_dataset.pickle", "rb").read())
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        self.MINIMUM_CONFIDENCE = 0.7
        self.MODELS = {
            "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
            "high-res-frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_fpn,
            "low-res-frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
            "retinanet": detection.retinanet_resnet50_fpn,
            "fcos-resnet": detection.fcos_resnet50_fpn,
            "ssd300": detection.ssd300_vgg16,
            "ssdlite-mobilenet": detection.ssdlite320_mobilenet_v3_large
        }
        self.model = None
        self.is_detection_successfully_performed = False

    def load(self, pretrained_model_name: str):
        self.model = self.MODELS[pretrained_model_name](pretrained=True,
                                                        progress=True,
                                                        num_classes=len(self.CLASSES),
                                                        pretrained_backbone=True).to(self.DEVICE)
        self.model.eval()

    def perform_detection_on(self, image):
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig = image.copy()

        image = image.transpose((2, 0, 1))

        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        image = torch.FloatTensor(image)

        image = image.to(self.DEVICE)
        detections = self.model(image)[0]

        for i in range(0, len(detections["boxes"])):

            confidence = detections["scores"][i]

            if confidence > self.MINIMUM_CONFIDENCE:

                idx = int(detections["labels"][i])

                if self.CLASSES[idx-1] != "car":
                    continue

                box = detections["boxes"][i].detach().cpu().numpy()
                (startX, startY, endX, endY) = box.astype("int")

                label = "{}: {:.2f}%".format(self.CLASSES[idx-1], confidence * 100)
                print("[INFO] {}".format(label))

                cv2.rectangle(orig, (startX, startY), (endX, endY),
                              self.COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(orig, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)

        base64_image = self.ndarray_to_base64(orig)

        self.is_detection_successfully_performed = True
        return self.is_detection_successfully_performed, base64_image

    def ndarray_to_base64(self,ndarray):
        img = cv2.cvtColor(ndarray, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode('utf-8')


