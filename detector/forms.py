from django import forms

class DetectionModelForm(forms.Form):
    CHOICES = [
               ('frcnn-resnet', 'Faster R-CNN - ResNet-50'),
               ('high-res-frcnn-mobilenet', 'Faster R-CNN - MobileNetV3-Large'),
               ('low-res-frcnn-mobilenet', 'Faster R-CNN - MobileNetV3-Large (mobile)'),
               ('fcos-resnet', 'FCOS - ResNet-50'),
               ('ssd300', 'SSD300'),
               ('ssdlite-mobilenet', 'SSDLite - MobileNetV3'),
               ('yolov5n', 'YOLOv5 (nano)'),
               ('yolov5s', 'YOLOv5 (small)'),
               ('yolov5m', 'YOLOv5 (medium)'),
               ('yolov5l', 'YOLOv5 (large)'),
               ('yolov5x', 'YOLOv5 (extra large)')]

    model = forms.ChoiceField(choices=CHOICES, widget=forms.RadioSelect)