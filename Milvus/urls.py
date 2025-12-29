from django.urls import path
from .views import CreateKBView


urlpatterns = [
  path('CreateKnowledgeBase/', CreateKBView.as_view(), name='create_knowledge_base'),
]