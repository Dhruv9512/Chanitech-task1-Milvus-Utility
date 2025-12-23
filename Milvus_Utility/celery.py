import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Milvus_Utility.settings')

app = Celery('Milvus_Utility')

app.config_from_object('django.conf:settings', namespace='CELERY')

app.autodiscover_tasks()
