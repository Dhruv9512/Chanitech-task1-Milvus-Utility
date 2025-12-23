from django.contrib import admin
from .models import (
    BaseKnowledge,
    KnowledgeBaseDetails,
    HtmlToMarkdownConversion,
    BaseHtmlToMarkdownConversion
)


@admin.register(BaseKnowledge)
class BaseKnowledgeAdmin(admin.ModelAdmin):
    list_display = (
        'batch_id',
        'domain',
        'user_uuid',
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
        'scan_id',
        'xml_id',
        'is_normal',
        'is_path_scan',
        'status_code',
        'total_words',
        'tokens',
        'created_at',
        'json_ld_schema',
    )

    search_fields = (
        'batch_id',
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



@admin.register(BaseHtmlToMarkdownConversion)
class BaseHtmlToMarkdownConversionAdmin(admin.ModelAdmin):
    # 🔹 What you see in admin list page
    list_display = (
        "id",
        "batch_id",
        "domain",
        "scan_id",
        "version",
        "scan_status",
        "markdown_conversion_url_count",
        "credit_utilized",
        "is_llm",
        "is_doc_gpt",
        "created_at",
    )

    # 🔹 Filters on right sidebar
    list_filter = (
        "scan_status",
        "category",
        "is_llm",
        "is_doc_gpt",
        "is_full_scan",
        "is_path_scan",
        "is_extension",
        "created_at",
    )

    # 🔹 Search box
    search_fields = (
        "batch_id",
        "domain",
        "user_uuid",
        "link_url_hash",
        "title",
    )

    # 🔹 Ordering
    ordering = ("-created_at",)

    # 🔹 Read-only fields (important IDs should not change)
    readonly_fields = (
        "batch_id",
        "task_id",
        "project_id",
        "created_at",
        "updated_at",
    )

    # 🔹 Performance optimization for big tables
    list_per_page = 50
    show_full_result_count = False

    # 🔹 Field grouping (clean UI)
    fieldsets = (
        ("Basic Info", {
            "fields": (
                "user_uuid",
                "domain",
                "category",
                "title",
                "scan_status",
                "note",
            )
        }),
        ("Batch & Scan", {
            "fields": (
                "scan_id",
                "version",
                "batch_id",
                "task_id",
                "project_id",
                "unique_project_id",
            )
        }),
        ("Counts & Credits", {
            "fields": (
                "markdown_total_summary_count",
                "markdown_conversion_url_count",
                "exclude_urls_count",
                "credit_utilized",
                "total_rag_credits",
                "batch_tokens",
            )
        }),
        ("Flags", {
            "fields": (
                "is_llm",
                "is_doc_gpt",
                "is_full_scan",
                "is_path_scan",
                "is_extension",
                "cleaned_html",
                "is_non_html",
                "enable_parallel",
            )
        }),
        ("Advanced JSON Fields", {
            "classes": ("collapse",),
            "fields": (
                "schema",
                "include_parents_path",
                "exclude_parents_path",
                "include_url_hash_list",
                "exclude_url_hash_list",
                "specific_ext",
                "sitemap_xml_include_path",
                "sitemap_xml_exclude_path",
                "link_url_list",
            )
        }),
        ("Timestamps", {
            "fields": (
                "created_at",
                "updated_at",
            )
        }),
    )
