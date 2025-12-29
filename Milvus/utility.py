from typing import List,Tuple
from urllib.parse import urlparse
import os
import re
import tiktoken
from markdownify import markdownify as md
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType,utility
from langchain_core.documents import Document
import mysql.connector


EMBEDDING_DIM_MAP = {
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    "intfloat/multilingual-e5-large": 1024,
    "gemini-text-embedding-004": 768,
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}

PRICING_MAP = {
    "text-embedding-3-small": 0.020,  
    "text-embedding-3-large": 0.130,  
    "ada-002": 0.100,                 
    "gemini-text-embedding-004": 0.00, 
    # For Local
    "intfloat/multilingual-e5-large": 0.0,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 0.0,
    "paraphrase-multilingual-MiniLM-L12-v2": 0.0,
}

# Function to calculate Token and proce as par the documents
def get_embedding_cost(docs: List[Document], model_name: str) -> Tuple[int, float]:
    total_tokens = 0

    if "text-embedding" in model_name or "openai" in model_name.lower():
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
            
        for doc in docs:
            total_tokens += len(encoding.encode(doc.page_content))
    else:
        for doc in docs:
            text_len = len(doc.page_content)
            total_tokens += text_len // 4
    # Get price per 1M tokens 
    price_per_1m = PRICING_MAP.get(model_name, 0.0)
    
    # Cost = (Tokens / 1,000,000) * Price
    total_cost = (total_tokens / 1_000_000) * price_per_1m

    return total_tokens, total_cost

# Function to create Milvus collection based on embedding model dimension
def create_collection(collection_name: str,embedding_model_name: str)-> None:
    """
    Create Milvus collection based on embedding model dimension.
    """
    if embedding_model_name not in EMBEDDING_DIM_MAP:
        raise ValueError(f"Unsupported embedding model: {embedding_model_name}")

    dim = EMBEDDING_DIM_MAP[embedding_model_name]


    # Drop existing collection
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    # Define collection schema
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),

        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),

        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=65535, is_nullable=True),
        FieldSchema(name="doc_url", dtype=DataType.VARCHAR, max_length=65535, is_nullable=True),
        FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=65535, is_nullable=True),
        FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=65535, is_nullable=True),
        FieldSchema(name="original_type", dtype=DataType.VARCHAR, max_length=65535, is_nullable=True),

        FieldSchema(name="page", dtype=DataType.INT64, is_nullable=True),
        FieldSchema(name="start_index", dtype=DataType.INT64, is_nullable=True),
        FieldSchema(name="doc_index", dtype=DataType.INT64, is_nullable=True),

        FieldSchema(name="parent_batch_id", dtype=DataType.VARCHAR, max_length=65535, is_nullable=True),
        FieldSchema(name="batch_id", dtype=DataType.VARCHAR, max_length=65535, is_nullable=True),
        FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=65535, is_nullable=True),
        FieldSchema(name="end_index", dtype=DataType.INT64, is_nullable=True),

        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="json_ld_schema", dtype=DataType.JSON,default=dict, blank=True, null=True)
    ]

    schema = CollectionSchema(
        fields=fields,
        description=f"RAG KB using {embedding_model_name} embeddings",
    )

    Collection(
        name=collection_name,
        schema=schema
    )

def clean_domain(domain: str):
    return re.sub(r'[^a-zA-Z0-9_]', '_', domain)

def clean_parent_batch_id(parent_batch_id: str):
    return str(parent_batch_id).replace('-', '_')

# Function to fetch data from HtmlToMarkdownConversion model based on batch_id
MYSQL_CONFIG={
    "host":os.getenv("MYSQL_HOST"),
    "user":os.getenv("MYSQL_USER"),
    "password":os.getenv("MYSQL_PASSWORD"),
    "database":os.getenv("MYSQL_DB_NAME"),
    "port":int(os.getenv("MYSQL_PORT", 3306))
}
def fetch_markdown(batch_id, skip, limit):
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    batch_id = str(batch_id).replace('-', '')
    query1 = f"""
    SELECT scan_id
    FROM link_scrapper_app_basehtmltomarkdownconversion
    WHERE batch_id = %s
    LIMIT 1;
    """
    print(f"scan query1: {query1} - {batch_id}")
    cursor.execute(query1, (batch_id,))
    scan_ids = cursor.fetchone()
    print(f"scan_ids: {scan_ids}")
    scan_id = scan_ids[0]

    if scan_id:
        query = f"""
        SELECT markdown_content, link_url, link_id, link_url_hash, xml_id, document_url,json_ld_schema
        FROM link_scrapper_app_htmltomarkdownconversion
        WHERE scan_id={scan_id} AND batch_id = %s
        LIMIT {skip}, {limit};
        """
    else:
        query = f"""
        SELECT markdown_content, link_url, link_id, link_url_hash, xml_id, document_url, json_ld_schema
        FROM link_scrapper_app_htmltomarkdownconversion
        WHERE scan_id is null AND batch_id = %s
        LIMIT {skip}, {limit};
        """

    # print(f"fetch query: {query} - {batch_id}")
    cursor.execute(query, (batch_id,))
    rows = cursor.fetchall()
    # print(f"ROWS: {rows}")
    markdown_contents = [row[0] for row in rows]
    urls = [row[1] for row in rows]
    link_ids = [row[2] for row in rows]
    link_url_hashes = [row[3] for row in rows]
    xml_ids = [row[4] for row in rows]
    s3_urls = [row[5] for row in rows]
    json_ld_schema=[row[6] for row in rows]
    conn.close()

    return markdown_contents, urls, link_ids, link_url_hashes, xml_ids, s3_urls,json_ld_schema
 


# Utility function to extract file extension from URL
def get_extension_from_url(url: str) -> str | None:
    path = urlparse(url).path          
    _, ext = os.path.splitext(path)   
    return ext.lower() if ext else None

# Function to convert HTML text to Markdown format
def fetch_and_convert_to_md(html_text: str) -> str:
    markdown_text = md(html_text)

    return markdown_text

