from torchvision.models import detection
import numpy as np
import pickle
import torch
import cv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = pickle.loads(open("coco_dataset.pickle", "rb").read())
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
MINIMUM_CONFIDENCE = 0.7

MODELS = {
    "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
    "high-res-frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_fpn,
    "low-res-frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    "retinanet": detection.retinanet_resnet50_fpn,
    "fcos-resnet": detection.fcos_resnet50_fpn,
    "ssd300": detection.ssd300_vgg16,
    "ssdlite-mobilenet": detection.ssdlite320_mobilenet_v3_large
}

model = MODELS["frcnn-resnet"](pretrained=True, progress=True,
                              num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
model.eval()

image = "human_car.jpg"
image = cv2.imread(image)
orig = image.copy()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.transpose((2, 0, 1))

image = np.expand_dims(image, axis=0)
image = image / 255.0
image = torch.FloatTensor(image)

image = image.to(DEVICE)
detections = model(image)[0]

for i in range(0, len(detections["boxes"])):

    confidence = detections["scores"][i]

    if confidence > MINIMUM_CONFIDENCE:

        idx = int(detections["labels"][i])

        if CLASSES[idx-1] != "car":
            continue

        box = detections["boxes"][i].detach().cpu().numpy()
        (startX, startY, endX, endY) = box.astype("int")

        label = "{}: {:.2f}%".format(CLASSES[idx-1], confidence * 100)
        print("[INFO] {}".format(label))

        cv2.rectangle(orig, (startX, startY), (endX, endY),
                      COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(orig, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

cv2.imshow("Output", orig)
cv2.waitKey(0)

