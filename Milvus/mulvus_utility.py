from sentence_transformers import SentenceTransformer
from typing import List, Dict
from langchain_core.documents import Document
from rest_framework.response import Response
from rest_framework import status
import os
import requests
import fitz
from dotenv import load_dotenv
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from contextlib import contextmanager
import pymysql
from pymilvus import connections,utility,Collection
import logging
from django.db.models import F
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Milvus
from .models import BaseKnowledge,KnowledgeBaseDetails
from .utility import fetch_data, get_extension_from_url,fetch_and_convert_to_md,create_collection,clean_domain,clean_parent_batch_id,get_embedding_cost

load_dotenv()
logger = logging.getLogger(__name__)
FETCH_LIMIT=50
CHUNK_SIZE=2000
CHUNK_OVERLAP=400
USER_RAG_CREADIT=50000
MILVUS_MAX_BATCH_SIZE=250


# Utility class for generating sentence embeddings using SentenceTransformer
class SentenceTransformerUtility:
    """
    Utility class that create local sentence Transformer object to use it as embedder
    """
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32):
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_model_name=model_name
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer model {model_name}: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            # sentence-transformers class method that embaddeing data that will be store in Vector DB
            embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            return []

    def embed_query(self, text: str) -> List[float]:
        # sentence-transformers class method that embaddeing user queary
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return []

# Making a Object of the SentenceTransformerUtility class with specific model and device
try:
    embedding_model = SentenceTransformerUtility(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
        batch_size=32  
    )
except Exception:
    embedding_model = None

# Utility Class to store embedding vectors data into Milvus
class MilvusUtility:
    """
    Utility Class to manage fetching, processing, chunking, and storing data into Milvus.
    """
    def __init__(self , parent_batch_id:str , domain:str , batch_id_list:List[str] , task_id:str , task_uuid:str , is_openai:bool):
        self.parent_batch_id = parent_batch_id
        self.domain = domain
        self.batch_id_list = batch_id_list
        self.task_id = task_id
        self.task_uuid = task_uuid
        self.is_openai = is_openai
        self.collection_name = f"collection_{clean_domain(self.domain)}_{clean_parent_batch_id(self.parent_batch_id)}"
        self.textsplitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP)
        self.token_cost=0
        self.token=0
        self.base_instance=None
        self.vectorstore=None

        # Create embedder object base on is_openai flag
        if self.is_openai:
            try:
                self.embedder = OpenAIEmbeddings(
                    openai_api_key=os.getenv("OPENAI_API_KEY"),
                    model="text-embedding-3-small"
                )

                self.embedding_model_name="text-embedding-3-small"

                self.embedder.embed_query("test")
            except Exception as e:
                self.embedder = HuggingFaceEndpointEmbeddings(
                    model="intfloat/multilingual-e5-large",
                    huggingfacehub_api_token=os.getenv("HuggingFace_API_KEY")
                )
                self.embedding_model_name="intfloat/multilingual-e5-large"
        else:
            self.embedder = embedding_model
            self.embedding_model_name=embedding_model.embedding_model_name
        
    # Builder function to create the milvus collection and insert the data
    def builder(self)->bool:

        """
        Main orchestration method:
        1. Initialize tracking.
        2. Connect to Milvus.
        3. Create Collection (Once).
        4. Process all batches.
        5. Cleanup.
        """
        try:
            # First assign new task_id and tsk_uuid to the Baseknowlege model and validate the instance
            self._initialize_base_knowledge_instance(self.task_id , self.task_uuid , self.domain , self.parent_batch_id)

            self._send_ws_update("Start Creating KB")
            # Connect Milvus
            self._create_milvus_connection()

            self._send_ws_update("Start Creating Collection")
            # Create Collection
            create_collection(self.collection_name , self.embedding_model_name)

            self._send_ws_update("Start Processing LLM ready Data")
            # Loop through all batch_id_list and fetch data and process,chunking it and store it into milvus 
            for batch_id in self.batch_id_list:
                # Fetch data of particular batch_id 
                try:
                    self._Process_Store_Batch(batch_id)
                    self._send_ws_update(f"Processing batch {batch_id}")
                except Exception as e:
                    logger.error(f"Error processing batch {batch_id}: {e}")
                    self._send_ws_update(f"Error processing batch {batch_id}: {e}")
                    continue
            self._update_embedding_costs(self.token,self.token_cost)
            # Cleanup the milvus collection and memory
            self._cleanup_milvus()

            self._send_ws_update("KB Created Successfully")
            return True
        except Exception as e:
                logger.critical(f"Critical error in builder: {e}")
                self._cleanup_milvus()
                self._send_ws_update(f"Error in builder: {e}")
                return False
        
    # Function to initialize the BaseKnowledge model instance
    def _initialize_base_knowledge_instance(self , task_id:str , task_uuid:str , domain:str , parent_batch_id:str)->None:
        try:
            self.base_instance = BaseKnowledge.objects.filter(
                domain=domain, 
                batch_id=parent_batch_id
            ).first()
            
            if self.base_instance:
                self.base_instance.task_id = task_id
                self.base_instance.save()
            else:
                logger.error(f"BaseKnowledge instance not found for domain {domain} and batch {parent_batch_id}")
        except Exception as e:
            logger.error(f"Failed to initialize BaseKnowledge instance: {e}")
    
    # Function to process and store data of particular batch_id
    def _Process_Store_Batch(self, batch_id: str) -> None:
        """
        Fetches, filters, processes, and stores data for a specific batch_id.
        """
        OFFSET=0
        fetch_limit=FETCH_LIMIT

        try:
            while True:
                # Fetch data from HtmlToMarkdownConversion model
                markdown_content,link_id,link_url_hash,xml_id,s3_url,urls = fetch_data(batch_id, offset=OFFSET, limit=fetch_limit)
                if not markdown_content:
                    break

                # Filtering the urls that alrady in KnowledgeBaseDetails
                filter_data=self._filter_existing_data(batch_id, markdown_content, link_id, link_url_hash, xml_id, s3_url, urls)
                
                if filter_data is True:
                    logger.info(f"Skipping offset {OFFSET}: All items already exist.")
                    OFFSET += fetch_limit
                    continue

                markdown_content, link_id, link_url_hash, xml_id, s3_url, urls = filter_data

                # Process the data from those fetched results
                MarkdownContents=self._process_data(markdown_content,link_id,link_url_hash,xml_id,s3_url,batch_id,urls)

                self._send_ws_update(f"Processing batch {batch_id} Successfully")
                # Chunk the data and make Documents
                results=self._Chunk_and_Make_Documents(MarkdownContents)
            
                # Insert the documents into milvus collection
                self._Insert_into_milvus_collection(results)

                self._send_ws_update(f"Inserting batch {batch_id} Successfully")
                # Make a KnowledgeBaseDetails entry for tracking progress
                self._KnowledgeBaseDetails_Entry(link_id,urls,link_url_hash,xml_id,batch_id)

                # Update and Calculate token and Price as par document
                token,token_cost=get_embedding_cost(results,self.embedding_model_name)
                self.token+=token
                self.token_cost+=token_cost

                OFFSET += fetch_limit
        except Exception as e:
            logger.error(f"Error in batch loop at offset {OFFSET}: {e}")
            OFFSET += fetch_limit 
       

    # Function to process the fetched data and Return the List of Documents
    def _process_data(self, markdown_content: List[str],link_id: List[str],link_url_hash: List[str],xml_id: List[str],s3_url: List[str], batch_id: str, urls: List[str]) -> List[List[Dict]]:
        
        if not markdown_content:
            return Response({'error': 'No data found for the given batch_id'}, status=status.HTTP_404_NOT_FOUND)
        
        MarkdownContents=[]
        for markdown, url, s3 in zip(markdown_content, urls, s3_url):
            try:
                ext = get_extension_from_url(url)
                default_metadata = {
                        "domain": self.domain,
                        "parent_batch_id": str(self.parent_batch_id),
                        "batch_id": batch_id,
                        "section": "",       
                        "start_index": -1, 
                        "doc_index": -1,    
                        "end_index": -1      
                    }
                if ext in ['.pdf']:
                    MarkdownContents.extend(self._get_pdf_data( url, s3))
                    self._send_ws_update(f"Process pdf data of batch {batch_id} Successfully")
                else:
                    MarkdownContents.extend(self._get_markdown_data(markdown, url, s3))
                    self._send_ws_update(f"Process markdown data of batch {batch_id} Successfully")
                for c in MarkdownContents:
                    c.setdefault("metadata", {}).update(default_metadata)
            except Exception as e:
                logger.error(f"Error processing item {url}: {e}")
                continue
        return MarkdownContents

    # Function to get text and metadata from PDF documents
    def _get_pdf_data(self, url: str, s3_url: str) -> List[Dict]:
        ans = []

        try:
            response = requests.get(s3_url, timeout=30)
            response.raise_for_status()
            pdf_bytes = response.content

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch PDF from S3: {s3_url} | Error: {e}")
            return []

        try:
            pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
            pdf_metadata = pdf.metadata or {}

            for page_number, page in enumerate(pdf, start=1):
                try:
                    text = page.get_text().strip()
                    if not text:
                        continue

                    metadata = {
                        "source_url": url,
                        "s3_url": s3_url,
                        "page_number": page_number,
                        "pdf_title": pdf_metadata.get("title", ""),
                        "pdf_author": pdf_metadata.get("author", ""),
                        "pdf_subject": pdf_metadata.get("subject", ""),
                        "pdf_keywords": pdf_metadata.get("keywords", ""),
                    }

                    ans.append({
                        "text": text,
                        "metadata": metadata
                    })

                except Exception as page_error:
                    logger.warning(
                        f"Failed to parse page {page_number} in PDF {s3_url}: {page_error}"
                    )
                    continue

            pdf.close()
        except fitz.FileDataError as e:
            logger.error(f"Invalid or corrupted PDF: {s3_url} | Error: {e}")
            return []
        except Exception as e:
            logger.exception(f"Unexpected error while processing PDF: {s3_url}")
            return []

        return ans
    
    # Function to get text and metadata from markdown content url
    def _get_markdown_data(self, markdown_content: List[str], url: str, s3_url: str) -> List[Dict]:
        try:
            if isinstance(markdown_content, str) and (
                markdown_content.startswith("http://") or markdown_content.startswith("https://")
            ):
                response = requests.get(s3_url, timeout=30)
                response.raise_for_status()
                text = fetch_and_convert_to_md(response.text)
            else:
                text = markdown_content

            metadata = {
                "source": url,
                "doc_url": s3_url if s3_url else url,
                "type": "markdown",
                "original_type": "md",
                "page": 1
            }

            return [{"text": text, "metadata": metadata}]

        except requests.exceptions.RequestException as e:
            logger.error(f"Markdown fetch failed: {s3_url} | {e}")
            return []

        except Exception as e:
            logger.exception("Unexpected error while processing markdown")
            return []
        
    # Function to chunk the data and make Documents
    def _Chunk_and_Make_Documents(self, MarkdownContents: List[Dict]) -> List[Document]:
        documents = []

        try:
            for item in MarkdownContents:
                text = item.get("text", "")
                metadata = item.get("metadata", {})

                if not text:
                    continue

                chunks = self.textsplitter.split_text(text)

                for chunk in chunks:
                    documents.append(
                        Document(page_content=chunk, metadata=metadata)
                    )

        except Exception as e:
            logger.exception("Error while chunking and creating documents")

        return documents

    # Function to insert the documents into milvus collection
    def _Insert_into_milvus_collection(self, documents: List[Document]) -> None:
        try:
            if not documents:
                return
            if len(documents)>MILVUS_MAX_BATCH_SIZE:
                self._Milvus_large_batch(documents)
                print("Large Batch")
            else:
                self._Milvus_small_batch(documents)
                print("Small Batch")
        except Exception as e:
            logger.exception("Error while inserting documents into Milvus")

    # Function to insert the documents into milvus collection in large batch
    def _Milvus_large_batch(self, documents: List[Document]) -> None:
        try:
            if not documents:
                return

            for i in range(0,len(documents),MILVUS_MAX_BATCH_SIZE):
                self._Milvus_small_batch(documents[i:i+MILVUS_MAX_BATCH_SIZE])
        except Exception as e:
            logger.exception("Error while inserting documents into Milvus")
               
    # Function to insert the documents into milvus collection in small batch
    def _Milvus_small_batch(self, documents: List[Document]) -> None:
        try:
            if not documents:
                return
            if self.vectorstore is None:
                self.vectorstore = self._check_create_vector_store(documents)
            else:
                self.vectorstore.add_documents(documents)

        except Exception as e:
            logger.exception("Error while inserting documents into Milvus")
               
    # Function to check and create vector store in milvus collection
    def _check_create_vector_store(self, documents: List[Document]) -> None:
        try:
            vectorstore = Milvus.from_documents(
                documents=documents,
                embedding=self.embedder,
                collection_name=self.collection_name,
                connection_args={
                    "host":os.getenv("HOST"),
                    "port":os.getenv("PORT")
                }
            )

            return vectorstore

        except Exception:
            logger.exception("Error while creating or checking vector store")
            return None
    
    # Create Milvus Connecter method
    def _create_milvus_connection(self):
        try:
            connections.connect(
                alias="default",
                host=os.getenv("HOST"),
                port=os.getenv("PORT")
            )
        except Exception:
            logger.exception("Failed to create Milvus connection")
    
   
    # Cleanup milvus collection and memory
    def _cleanup_milvus(self):
        try:
            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)

                # Safe even if not loaded
                collection.release()

                logger.info(
                    f"Milvus collection unloaded: {self.collection_name}"
                )

        except Exception:
            logger.warning(
                "Milvus cleanup warning",
                exc_info=True
            )

        finally:
            try:
                connections.disconnect("default")
            except Exception:
                logger.warning(
                    "Milvus disconnect warning",
                    exc_info=True
                )

    # Create Mysql Connecter method
    @contextmanager
    def _get_mysql_connection(self):
        connection = None
        cursor = None
        try:
            connection = pymysql.connect(
                host=os.getenv("MYSQL_HOST"),
                user=os.getenv("MYSQL_USER"),
                password=os.getenv("MYSQL_PASSWORD"),
                database=os.getenv("MYSQL_DB_NAME"),
                port=int(os.getenv("MYSQL_PORT", 3306)),
                cursorclass=pymysql.cursors.DictCursor
            )
            cursor = connection.cursor()
            yield connection, cursor
        except Exception:
            logger.exception("MySQL connection error")
            raise
        finally:
            try:
                if cursor:
                    cursor.close()
                if connection:
                    connection.close()
            except Exception:
                pass

    # Method to do entries in Kn
    def _KnowledgeBaseDetails_Entry(self, link_ids: List, urls: List[str], url_hashes: List[str], xml_ids: List, batch_id: str) -> int:
        try:
            entries = []

            for link_id, url, url_hash, xml_id in zip(link_ids, urls, url_hashes, xml_ids):
                entries.append(
                    KnowledgeBaseDetails(
                        link_id=link_id,
                        link_url=url,
                        link_url_hash=url_hash,
                        parent_batch_id=self.parent_batch_id,
                        batch_id=batch_id,
                        xml_id=xml_id,
                        milvus_collection_name=self.collection_name,
                        file_name=None
                    )
                )
            if entries:
                KnowledgeBaseDetails.objects.bulk_create(entries, ignore_conflicts=True)
            return len(entries)
        except Exception:
            logger.exception("Error while creating KnowledgeBaseDetails entries")
            return 0

    # Update and Calculate the token and price as par document
    def _update_embedding_costs(self,total_tokens,embedding_cost)->None:
        global USER_RAG_CREADIT
        try:
            if total_tokens == 0 and embedding_cost == 0:
                logger.warning("No tokens or cost calculated, skipping update")
                return

            BaseKnowledge.objects.filter(
                id=self.base_instance.id
            ).update(
                total_embedding_token=total_tokens,
                total_embedding_cost=embedding_cost
            )

            logger.info(
                f"Updated embedding cost +{embedding_cost}, tokens +{total_tokens}"
            )

            if USER_RAG_CREADIT > 0:
                USER_RAG_CREADIT-=embedding_cost

        except Exception as e:
            # Log error but don't crash the pipeline just for metrics
            logging.error(f"Failed to update costs: {e}")


    # Filtering the urls that alrady in KnowledgeBaseDetails
    def _filter_existing_data(self, batch_id: str, markdown_content: list, link_id: list, link_url_hash: list, xml_id: list, s3_url: list, urls: list):

        try:
            existing_hashes = set(
                KnowledgeBaseDetails.objects.filter(batch_id=batch_id,parent_batch_id=self.parent_batch_id)
                .values_list("link_url_hash", flat=True)
            )

            if not existing_hashes:
                self._send_ws_update(f"No existing data found for batch {batch_id}")
                return markdown_content, link_id, link_url_hash, xml_id, s3_url, urls

            zipped_data = zip(
                markdown_content,
                link_id,
                link_url_hash,
                xml_id,
                s3_url,
                urls
            )

            filtered_rows = [
                row for row in zipped_data if row[2] not in existing_hashes
            ]

            if not filtered_rows:
                self._send_ws_update(f"No new data found for batch {batch_id}")
                return True

            return map(list, zip(*filtered_rows))

        except Exception:
            logger.exception("Error while filtering existing knowledge base data")
            return True
    

    # Send the updates using websocket
    def _send_ws_update(self,text:str):
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f"task_{self.task_id}",
            {
                "type": "task_update",
                "data": {"Status":text}
            }
        )
    # Function to create new object of this class and call the builder function
    def create_and_build(self) -> bool:
        try:
            return self.builder()
        except Exception:
            logger.exception("Error while creating and building knowledge base")
            return False