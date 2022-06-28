from django import forms

class DetectionModelForm(forms.Form):
    CHOICES = [('yolov5n', 'YOLOv5 (nano)'),
               ('yolov5s', 'YOLOv5 (small)'),
               ('yolov5m', 'YOLOv5 (medium)'),
               ('yolov5l', 'YOLOv5 (large)')]

    model = forms.ChoiceField(choices=CHOICES, widget=forms.RadioSelect)