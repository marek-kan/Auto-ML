from django.urls import path
from .views import classification

urlpatterns = [
    path('', classification, name='classification')
]
