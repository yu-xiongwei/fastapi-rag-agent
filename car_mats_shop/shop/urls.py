from django.contrib import admin
from django.urls import path
from apps.views import home, browse  # 加 browse

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='home'),
    path('browse/', browse, name='browse'),  # 新增
]