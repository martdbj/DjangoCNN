from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt

from .apps import AppConfig
import os

@csrf_exempt
def index_view(request):
    return render(request, 'index.html')

@csrf_exempt
def upload_view(request):
    context = {
        'prediction': None,
        'confidence': None,
        'image_url': None
    }

    if request.method == 'POST' and request.FILES.get('image'):
        img_file = request.FILES['image']

        fileSystem = FileSystemStorage()
        filename = fileSystem.save(img_file.name, img_file)
        uploaded_file_path = fileSystem.path(filename)
        uploaded_file_url = fileSystem.url(filename)

        try:
            label, score = AppConfig.predictor.predict(uploaded_file_path)

            context['prediction'] = label
            context['confidence'] = score
            context['image_url'] = uploaded_file_url

        except Exception as e:
            print(f"Error: {e}")
            context['prediction'] = "Prediction failed"

    return render(request, 'index.html', context)