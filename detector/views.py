from django.shortcuts import render


def file_uploader_view(request):
    return render(request, "detector/file_uploader.html")