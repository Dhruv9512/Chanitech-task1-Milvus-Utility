from django.db import models
import uuid


# Create your models here.
class BaseKnowledge(models.Model):
    user_uuid = models.CharField(max_length=40, null=True, blank=True)
    task_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True, null=True, blank=True)
    batch_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    domain = models.CharField(max_length=100, null=True, blank=True)
    batch_id_list = models.JSONField(default=list, blank=True)
    link_url_list = models.JSONField(default=list, blank=True)
    visitor_id = models.CharField(max_length=50, null=True, blank=True)
    total_count = models.IntegerField(default=0)
    category = models.CharField(max_length=50, null=True, blank=True)
    credit_utilized = models.IntegerField(default=0)
    url_count = models.IntegerField(default=0)
    scan_status = models.CharField(max_length=50, null=True, blank=True)
    milvus_collection_name = models.CharField(max_length=60, null=True, blank=True)
    embeded_model_name = models.CharField(max_length=100, null=True, blank=True, default='text-embedding-3-large')
    s3_key = models.CharField(max_length=500, null=True, blank=True)
    note = models.TextField(null=True, blank=True)
    title = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    total_embedding_cost = models.FloatField(default=0)
    total_embedding_token = models.IntegerField(default=0)
    last_inserted_id = models.BigIntegerField(default=0)
    extension_agent = models.CharField(max_length=100, null=True, blank=True)
    class Meta:
        indexes = [
            models.Index(fields=['batch_id', 'domain']),
            models.Index(fields=['batch_id', 'visitor_id', 'user_uuid']),
            models.Index(fields=['batch_id', 'user_uuid', 'visitor_id', 'domain']),
            models.Index(fields=['batch_id', 'domain', 'visitor_id']),
            models.Index(fields=['batch_id','domain', 'user_uuid']),
            models.Index(fields=['extension_agent', 'user_uuid']),
            models.Index(fields=['user_uuid', 'extension_agent'])
        ]
        db_table = 'Milvus_baseknowledge'


class KnowledgeBaseDetails(models.Model):
    link_id = models.IntegerField(null=True, blank=True)
    link_url = models.URLField(max_length=1000, null=True, blank=True)
    link_url_hash = models.CharField(max_length=256, null=True, blank=True)
    parent_batch_id = models.UUIDField(null=True, blank=True)
    batch_id = models.UUIDField(null=True, blank=True)
    xml_id = models.IntegerField(default=0)
    s3_url_key = models.CharField(max_length=1024, null=True, blank=True)
    milvus_collection_name = models.CharField(max_length=60, null=True, blank=True)
    error_due_to = models.TextField(null=True, blank=True)
    file_name = models.CharField(max_length=512, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=['parent_batch_id'])
        ]
        db_table = 'Milvus_knowledgebasedetails'


class HtmlToMarkdownConversion(models.Model):
    batch_id = models.UUIDField(null=True, blank=True, default=uuid.uuid4)
    visitor_id = models.CharField(max_length=255)

    link_id = models.IntegerField(null=True, blank=True)
    scan_id = models.IntegerField(null=True, blank=True)
    link_url = models.CharField(max_length=1000, null=True, blank=True)
    link_url_hash = models.CharField(max_length=8, null=True, blank=True)
    xml_id = models.IntegerField(default=0)
    is_normal=models.BooleanField(default=False)
    is_path_scan=models.BooleanField(default=False)

    markdown_content = models.TextField(null=True, blank=True)
    s3_url = models.CharField(max_length=500, null=True, blank=True)
    document_url = models.CharField(max_length=500, null=True, blank=True)
    onlymainContent = models.BooleanField(default=False)
    status_code = models.IntegerField(null=True, blank=True)
    error_due_to = models.TextField(null=True, blank=True)

    total_words = models.IntegerField(null=True, blank=True,default=0)
    total_character = models.IntegerField(null=True, blank=True,default=0)
    total_character_without_space = models.IntegerField(null=True, blank=True,default=0)
    tokens = models.IntegerField(null=True, blank=True,default=0)
    rag_credits = models.IntegerField(null=True, blank=True,default=0)

    internal_urls=models.JSONField(default=list, blank=True, null=True)
    internal_urls_count = models.IntegerField(default=0)
    external_urls=models.JSONField(default=list, blank=True, null=True)
    external_urls_count = models.IntegerField(default=0)

    json_ld_schema_exist = models.BooleanField(default=False)
    json_ld_schema = models.JSONField(default=dict, blank=True, null=True)

    schema = models.JSONField(default=dict, blank=True, null=True)
    detect_lang_details = models.JSONField(default=dict, blank=True, null=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.link_url} -> {self.visitor_id}"

    class Meta:
        db_table = "link_scrapper_app_htmltomarkdownconversion"
        unique_together = ('scan_id','xml_id', 'link_url_hash', 'batch_id')
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=["batch_id"], name="batch_id_idx_markdown_1"),
            models.Index(fields=["scan_id", "batch_id"], name="scan_id_batch_id_idx_1"),
            models.Index(fields=["scan_id", "visitor_id", "batch_id"], name="scan_visi_batch_id_idx_1"),
            models.Index(fields=["link_url_hash"], name="link_url_hash_idx_1"),
            models.Index(fields=["scan_id", "is_normal", "link_url_hash"], name="scan_id_link_hash_idx_1"),
        ]