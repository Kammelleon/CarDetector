import base64

from django.core.files import File
from django.shortcuts import render, redirect
import cv2
import numpy as np
from django.views import View

from .forms import DetectionModelForm

class FileUploader(View):
    uploaded_image = None
    base64_image = None

    def get(self, request):
        return render(request, "detector/file_uploader.html")

    def post(self, request):
        file = request.FILES["file"]
        numpy_converted_file = np.fromstring(file.read(), np.uint8)
        FileUploader.uploaded_image = cv2.imdecode(numpy_converted_file, cv2.IMREAD_UNCHANGED)
        if FileUploader.uploaded_image is None:
            # TODO: Add django message that file is not an image
            # TODO: Redirect back to upload site
            print("It is not an image")
        else:
            return redirect('detector:model-chooser')

class ModelChooser(View):
    selected_model = None

    def get(self,request):
        detection_model_form = DetectionModelForm()
        context = {
            "detection_model_form": detection_model_form
        }
        return render(request, "detector/model_chooser.html", context=context)

    def post(self, request):
        detection_model_form = DetectionModelForm(request.POST)
        if detection_model_form.is_valid():
            selected_model = detection_model_form.cleaned_data['model']
            ModelChooser.selected_model = selected_model
            # TODO: Deal with selected model
            if "yolov5" in selected_model:
                import detector.yolo
                detector.yolo.hi()

            return redirect("detector:image-previewer")

class ImagePreviewer(View):
    def get(self,request):
        return render(request, "detector/image_previewer.html")
