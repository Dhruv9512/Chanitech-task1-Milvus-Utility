from celery import shared_task
from .mulvus_utility import MilvusUtility


@shared_task(bind=True)
def create_and_build_task(self,**kwargs) -> bool:
    builder = MilvusUtility(
        parent_batch_id=kwargs.get("parent_batch_id"),
        domain=kwargs.get("domain"),
        batch_id_list=kwargs.get("batch_id_list"),
        task_id=self.request.id,
        task_uuid=kwargs.get("task_uuid"),
        is_openai=kwargs.get("is_openai")
    )
    return builder.create_and_build()
