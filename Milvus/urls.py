from django.urls import path
from .views import ImportConversionDataView, CreateKBView


urlpatterns = [
  path('InsertConversionData/', ImportConversionDataView.as_view(), name='insert_conversion_data'),
  path('CreateKnowledgeBase/', CreateKBView.as_view(), name='create_knowledge_base'),
]