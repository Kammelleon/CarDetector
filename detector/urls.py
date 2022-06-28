from django.urls import path
from detector.views import file_uploader_view

app_name = "detector"

urlpatterns = [
    path('', file_uploader_view, name="file-uploader"),
]
