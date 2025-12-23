from django.contrib import admin
from .models import (
    BaseKnowledge,
    KnowledgeBaseDetails,
    HtmlToMarkdownConversion
)


@admin.register(BaseKnowledge)
class BaseKnowledgeAdmin(admin.ModelAdmin):
    list_display = (
        'batch_id',
        'domain',
        'user_uuid',
        'visitor_id',
        'category',
        'scan_status',
        'total_count',
        'credit_utilized',
        'created_at',
    )

    search_fields = (
        'batch_id',
        'domain',
        'user_uuid',
        'visitor_id',
        'milvus_collection_name',
        'title',
    )

    list_filter = (
        'category',
        'scan_status',
        'created_at',
    )

    readonly_fields = (
        'task_id',
        'batch_id',
        'created_at',
        'updated_at',
        'total_embedding_cost',
        'total_embedding_token',
    )

    ordering = ('-created_at',)


@admin.register(KnowledgeBaseDetails)
class KnowledgeBaseDetailsAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'parent_batch_id',
        'batch_id',
        'link_id',
        'xml_id',
        'file_name',
        'created_at',
    )

    search_fields = (
        'parent_batch_id',
        'batch_id',
        'link_url',
        'file_name',
        'milvus_collection_name',
    )

    list_filter = (
        'created_at',
    )

    readonly_fields = (
        'created_at',
        'updated_at',
    )

    ordering = ('-created_at',)


@admin.register(HtmlToMarkdownConversion)
class HtmlToMarkdownConversionAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'batch_id',
        'visitor_id',
        'scan_id',
        'xml_id',
        'is_normal',
        'is_path_scan',
        'status_code',
        'total_words',
        'tokens',
        'created_at',
    )

    search_fields = (
        'batch_id',
        'visitor_id',
        'link_url',
        'link_url_hash',
        'document_url',
    )

    list_filter = (
        'is_normal',
        'is_path_scan',
        'json_ld_schema_exist',
        'created_at',
    )

    readonly_fields = (
        'created_at',
        'updated_at',
        'total_words',
        'total_character',
        'total_character_without_space',
        'tokens',
        'rag_credits',
    )

    ordering = ('-created_at',)
