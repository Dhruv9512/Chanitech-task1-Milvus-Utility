import csv
import json
import io
import uuid
import xml.etree.ElementTree as ET
from typing import List, Any,Tuple
from django.db import transaction
from django.core.exceptions import ValidationError
from .models import HtmlToMarkdownConversion
from urllib.parse import urlparse
import os
import re
import tiktoken
from markdownify import markdownify as md
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType,utility
from langchain_core.documents import Document
import gzip
import py7zr
from django.db import transaction
import chardet
from django.core.files.uploadedfile import UploadedFile
from bs4 import BeautifulSoup
import mysql.connector
import tempfile

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


# General Utility Class for HTML to Markdown Conversion Data Handling
class HtmlToMarkdownUtil:
    """
    General Utility Class for HTML to Markdown Conversion Data Handling
    Supports: CSV, XML (Excel XML & SQL Dump), XML.GZ, 7Z
    """

    # =========================================================
    # ENTRY POINT
    # =========================================================
    def create_and_insert(self, uploaded_file: UploadedFile) -> List[HtmlToMarkdownConversion]:
        if not hasattr(uploaded_file, "read"):
            raise TypeError(f"Expected UploadedFile, got {type(uploaded_file)}")

        filename = (uploaded_file.name or "").lower()

        if filename.endswith(".7z"):
            return self._process_7z(uploaded_file)

        if filename.endswith(".gz"):
            uploaded_file.seek(0)
            with gzip.open(uploaded_file, "rb") as f:
                raw = f.read()
            return self._route_content(self._decode_bytes(raw))

        uploaded_file.seek(0)
        raw = uploaded_file.read()
        return self._route_content(self._decode_bytes(raw))

    # =========================================================
    # ROUTER
    # =========================================================
    def _route_content(self, content: str):
        clean = content.strip().lstrip("\ufeff")
        if clean.startswith("<"):
            return self._process_xml(clean)
        return self._process_csv(clean)

    # =========================================================
    # CSV PARSER
    # =========================================================
    def _process_csv(self, csv_content: str) -> List[HtmlToMarkdownConversion]:
        reader = csv.reader(io.StringIO(csv_content))
        rows = list(reader)

        records = []
        for row in rows[1:]:
            if not row:
                continue
            try:
                records.append(self._map_data_to_model(row))
            except Exception as e:
                print(f"CSV Mapping Warning: {e}")
                continue

        return self._bulk_save(records)

    # =========================================================
    # XML PARSER
    # =========================================================
    def _process_xml(self, xml_content: str) -> List[HtmlToMarkdownConversion]:
        # Parse structure using the robust method
        rows = self._parse_xml_structure(xml_content)
        print(f"INFO: Parsed {len(rows)} rows from XML.")

        records = []
        # Skip header row (index 0) ONLY if it looks like a header (optional logic)
        # Note: Your SQL dump format usually doesn't have a header row in the data, 
        # but if the first row fails mapping, we just skip it.
        start_index = 0
        
        for i, row in enumerate(rows[start_index:]):
            if not row:
                continue
            try:
                records.append(self._map_data_to_model(row))
            except Exception:
                # Silently skip mapping errors for cleaner logs, or print if needed
                continue

        return self._bulk_save(records)

    # =========================================================
    # 7Z PARSER
    # =========================================================
    def _process_7z(self, uploaded_file: UploadedFile) -> List[HtmlToMarkdownConversion]:
        records = []
        uploaded_file.seek(0)
        file_stream = io.BytesIO(uploaded_file.read())

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with py7zr.SevenZipFile(file_stream, mode="r") as archive:
                    archive.extractall(path=temp_dir)
            except Exception as e:
                print(f"Error extracting 7z: {e}") 
                return []
            
            for root_dir, _, files in os.walk(temp_dir):
                for filename in files:
                    if not filename.lower().endswith(".xml"):
                        continue
                    
                    file_path = os.path.join(root_dir, filename)
                    try:
                        with open(file_path, 'rb') as f:
                            raw = f.read()
                            xml_content = self._decode_bytes(raw)
                            records.extend(self._process_xml(xml_content))
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                        continue
        return records

    # =========================================================
    # ROBUST XML STRUCTURE PARSER
    # =========================================================
    def _parse_xml_structure(self, xml_string: str) -> List[List[Any]]:
        """
        Parses XML using BeautifulSoup.
        Handles both Excel XML (<Cell><Data>) and SQL Dump XML (<field>).
        """
        # 1. Clean control characters
        xml_string = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', xml_string)
        
        rows = []
        try:
            # Use html.parser for maximum tolerance
            soup = BeautifulSoup(xml_string, "html.parser")
            
            # Find all "Row" tags (Excel) or "row" tags (SQL Dump)
            all_rows = soup.find_all(lambda tag: tag.name and tag.name.lower().endswith("row"))
            
            if len(all_rows) > 0:
                for xml_row in all_rows:
                    rows.append(self._extract_row_data_bs4(xml_row))
                return rows
        except Exception as e:
            print(f"XML Parsing Failed: {e}")

        # Fallback: Regex extraction if BS4 failed completely
        if not rows:
            print("WARNING: Switching to Regex Fallback for XML parsing.")
            row_pattern = re.compile(r'<[^:>]*:?Row[^>]*>(.*?)</[^:>]*:?Row>', re.DOTALL | re.IGNORECASE)
            # Also try lowercase 'row' for SQL dumps
            if not row_pattern.search(xml_string):
                 row_pattern = re.compile(r'<row>(.*?)</row>', re.DOTALL | re.IGNORECASE)

            raw_rows = row_pattern.findall(xml_string)
            for raw_row_content in raw_rows:
                # Regex for Data/Field tags
                data_pattern = re.compile(r'<[^:>]*:?(?:Data|field)[^>]*>(.*?)</[^:>]*:?(?:Data|field)>', re.DOTALL | re.IGNORECASE)
                cells_data = data_pattern.findall(raw_row_content)
                cleaned_row = [self._unescape_xml(c) for c in cells_data]
                rows.append(cleaned_row)

        return rows

    def _extract_row_data_bs4(self, xml_row) -> List[Any]:
        """Helper that handles both Excel XML and SQL Dump XML formats"""
        row_data = []
        
        # 1. Try finding Excel XML Cells (<Cell><Data>...</Data></Cell>)
        cells = xml_row.find_all(lambda tag: tag.name and tag.name.lower().endswith("cell"), recursive=False)
        
        if cells:
            # --- HANDLE EXCEL XML FORMAT ---
            for cell in cells:
                idx_attr = None
                for k, v in cell.attrs.items():
                    if k.lower().endswith("index"):
                        idx_attr = v
                        break
                
                if idx_attr:
                    try:
                        target_index = int(float(idx_attr)) - 1
                        while len(row_data) < target_index:
                            row_data.append(None)
                    except ValueError:
                        pass
                
                data_tag = cell.find(lambda tag: tag.name and tag.name.lower().endswith("data"))
                row_data.append(data_tag.get_text(strip=False) if data_tag else None)
        
        else:
            # --- HANDLE SQL DUMP FORMAT (<field>...</field>) ---
            # Your logs showed: <field name="id">...</field>
            fields = xml_row.find_all(lambda tag: tag.name and tag.name.lower() == "field", recursive=False)
            for field in fields:
                row_data.append(field.get_text(strip=False))
        
        return row_data

    def _unescape_xml(self, text):
        return (text.replace('&lt;', '<')
                    .replace('&gt;', '>')
                    .replace('&amp;', '&')
                    .replace('&quot;', '"')
                    .replace('&apos;', "'"))

    # =========================================================
    # CORE MAPPING LOGIC
    # =========================================================
    def _map_data_to_model(self, row: List[Any]) -> HtmlToMarkdownConversion:
        get_str = lambda i: self._clean_str(row, i)
        get_int = lambda i: self._clean_int(row, i)
        get_bool = lambda i: self._clean_bool(row, i)
        get_json = lambda i, d: self._clean_json(row, i, d)
        get_uuid = lambda i: self._clean_uuid(row, i)

        # IMPORTANT: Ensure your XML columns match this order!
        # Based on your logs: 0=id, 1=batch_id, 2=visitor_id, 3=link_id, etc.
        return HtmlToMarkdownConversion(
            batch_id=get_uuid(1),
            visitor_id=get_str(2),
            link_id=get_int(3),
            scan_id=get_int(4),
            link_url=get_str(5),
            link_url_hash=get_str(6),
            xml_id=get_int(7),
            is_normal=get_bool(8),
            markdown_content=get_str(9),
            status_code=get_int(10),
            error_due_to=get_str(11),
            is_path_scan=get_bool(14),
            onlymainContent=get_bool(15),
            s3_url=get_str(16),
            document_url=get_str(17),
            tokens=get_int(18),
            total_character=get_int(19),
            total_character_without_space=get_int(20),
            total_words=get_int(21),
            external_urls=get_json(22, list),
            external_urls_count=get_int(23),
            internal_urls=get_json(24, list),
            internal_urls_count=get_int(25),
            rag_credits=get_int(26),
            json_ld_schema=get_json(27, dict),
            json_ld_schema_exist=get_bool(28),
            schema=get_json(29, dict),
            detect_lang_details=get_json(30, dict),
        )

    # =========================================================
    # BULK SAVE
    # =========================================================
    def _bulk_save(self, records: List[HtmlToMarkdownConversion]):
        if records:
            with transaction.atomic():
                HtmlToMarkdownConversion.objects.bulk_create(
                    records,
                    ignore_conflicts=True
                )
        return records

    # =========================================================
    # SANITIZERS
    # =========================================================
    def _get_raw(self, row, index):
        try:
            val = row[index]
            if val in (None, "NULL", "null"):
                return None
            return val
        except IndexError:
            return None

    def _clean_str(self, row, index):
        val = self._get_raw(row, index)
        return str(val) if val is not None else None

    def _clean_int(self, row, index):
        val = self._get_raw(row, index)
        try:
            return int(float(val))
        except Exception:
            return 0

    def _clean_bool(self, row, index):
        val = self._get_raw(row, index)
        try:
            return bool(int(val))
        except Exception:
            return str(val).lower() == "true"

    def _clean_uuid(self, row, index):
        val = self._get_raw(row, index)
        try:
            return uuid.UUID(str(val))
        except Exception:
            return None

    def _clean_json(self, row, index, default):
        val = self._get_raw(row, index)
        if not val:
            return default() if callable(default) else default
        try:
            return json.loads(val)
        except Exception:
            try:
                fixed = (str(val).replace("'", '"')
                                 .replace("None", "null")
                                 .replace("True", "true")
                                 .replace("False", "false"))
                return json.loads(fixed)
            except Exception:
                return default() if callable(default) else default

    def _decode_bytes(self, raw_bytes: bytes) -> str:
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding") or "utf-8"
        try:
            return raw_bytes.decode(encoding, errors="replace")
        except Exception:
            return raw_bytes.decode("utf-8", errors="replace")