from django.core.files import File
from django.shortcuts import render
import cv2
import numpy as np

def file_uploader_view(request):
    if request.method == "POST":
        file = request.FILES["file"]
        numpy_converted_file = np.fromstring(file.read(), np.uint8)
        image = cv2.imdecode(numpy_converted_file, cv2.IMREAD_UNCHANGED)
        if image is None:
            print("It is not an image")
        else:
            print("It is image")
    return render(request, "detector/file_uploader.html")