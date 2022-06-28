from django.core.files import File
from django.shortcuts import render, redirect
import cv2
import numpy as np
from .forms import DetectionModelForm


def file_uploader_view(request):
    if request.method == "POST":
        file = request.FILES["file"]
        numpy_converted_file = np.fromstring(file.read(), np.uint8)
        image = cv2.imdecode(numpy_converted_file, cv2.IMREAD_UNCHANGED)
        if image is None:
            #TODO: Add django message that file is not an image
            #TODO: Redirect back to upload site
            print("It is not an image")
        else:
            return redirect('detector:model-chooser')
    return render(request, "detector/file_uploader.html")


def model_chooser_view(request):
    if request.method == "POST":
        detection_model_form = DetectionModelForm(request.POST)
        if detection_model_form.is_valid():
            selected_model = detection_model_form.cleaned_data['model']
            #TODO: Deal with selected model
    detection_model_form = DetectionModelForm()
    context = {
        "detection_model_form": detection_model_form
    }
    return render(request, "detector/model_chooser.html", context=context)
