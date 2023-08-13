# your_project_name/urls.py
from django.urls import path
from .views import *

urlpatterns = [
    path('', upload_files_view, name='upload_files'),
    path('custom_function/', custom_function, name='custom_function'),

    # Other URL patterns for your project
]
