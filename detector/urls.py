from django.urls import path
from detector.views import ImageUploader, ModelChooser, ImagePreviewer

app_name = "detector"

urlpatterns = [
    path('', ImageUploader.as_view(), name="image-uploader"),
    path('model/', ModelChooser.as_view(), name="model-chooser"),
    path('preview/', ImagePreviewer.as_view(), name="image-previewer"),
]
