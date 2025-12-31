"""
URL configuration for the KB application.
"""
from django.urls import path
from .views import CreateAndBuildKB
urlpatterns = [
    path('<str:batch_id>', CreateAndBuildKB.as_view(), name='create_and_build_KB'),
]
