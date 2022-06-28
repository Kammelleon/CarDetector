from django.urls import path
from detector.views import FileUploader, ModelChooser, ImagePreviewer

app_name = "detector"

urlpatterns = [
    path('', FileUploader.as_view(), name="file-uploader"),
    path('model/', ModelChooser.as_view(), name="model-chooser"),
    path('preview/', ImagePreviewer.as_view(), name="image-previewer"),
]
