from django.shortcuts import render
from apps.models import CarMat

def home(request):
    mats = CarMat.objects.all()
    return render(request, 'index.html', {'mats': mats})

# 新增：浏览页面（纯静态）
def browse(request):
    mats = CarMat.objects.all()
    return render(request, 'browse.html', {'mats': mats})