from django.shortcuts import render

# Create your views here.
def file_uploader_view(request):
    return render(request, "detector/file_uploader.html")