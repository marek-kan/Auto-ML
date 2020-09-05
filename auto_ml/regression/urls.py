from django.urls import path
from .views import regression

urlpatterns = [
    path('', regression, name='regression')
]
