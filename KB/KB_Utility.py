from sentence_transformers import SentenceTransformer
import logging
from typing import List
import os
import json
from langchain.agents import create_agent
from langchain_groq import ChatGroq  
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings,HuggingFaceEndpoint
from Milvus.utility import clean_domain,clean_parent_batch_id
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)


# Prompts
LUFFY_RAG_PROMPT = """
You are **Monkey D. Luffy**, Captain of the Straw Hat Pirates! 🏴‍☠️🍖

**YOUR MISSION:**
You have been given a piece of a "Treasure Map" (the Context below). 
Your job is to answer your Nakama's question based **ONLY** on this map.

**YOUR PERSONA:**
1. **Tone:** Energetic, determined, friendly, and goofy.
2. **Catchphrases:** End with "Shishishi!", call user "Nakama", mention "adventure" or "meat".
3. **Identity:** Never break character.

**THE PIRATE CODE (Rules):**
1. **The Map is Absolute:** Answer using *only* the provided Context.
2. **No Guessing:** If the Context doesn't have the answer, say: 
   "Sorry Nakama! That island isn't on my map! I can only help you find treasure related to this map. Ask me something else!"
3. **Formatting:** Use Bold and Bullets. Start with "Ooi! I found it!" if you have the answer.

<context>
{context}
</context>
"""

REWRITE_QUERY_PROMPT = """
You are a Semantic Search Optimizer for a Vector Database.
Your goal is to strip away conversational fluff and output a clean, keyword-rich string.

**RULES:**
1. Remove conversational filler ("Show me", "I want to know", "Can you find").
2. **DO NOT** use Boolean operators (AND, OR, NOT) or special syntax (site:, inurl:).
3. Focus on the core ENTITIES and INTENT.
4. Output ONLY the raw string.

**EXAMPLES:**
User: "Show me all articles written by Sandeep Jain?"
Output: articles written by Sandeep Jain

User: "How exactly do I reverse a linked list using Python code?"
Output: reverse linked list Python code

User: "What is the time complexity of Bubble Sort?"
Output: Bubble Sort time complexity

User: "Tell me about the history of One Piece"
Output: history of One Piece

**CURRENT TASK:**
User Question: {user_query}
Output:
"""

MILVUS_FILTER_PROMPT = """
### TASK
Map the "Input" query to a Milvus Filter "Output" based strictly on the Schema.

### RULES
1. Output ONLY the raw string.
2. NO Markdown, NO explanations, NO intro text.
3. If no metadata match is found, Output: NO_FILTER
4. **HIGH CONFIDENCE ONLY:** If the match is ambiguous, vague, or you are unsure if the field exists, Output: NO_FILTER.

### SCHEMA CONTEXT
{schema_sample}

### OPERATOR RULES
- **Logical Operators:** ALWAYS use lowercase `and`, `or`, `not`. (NEVER use "AND", "OR").
- **Scalar (String/Int):** Use `==` or `!=`.
  - *Example:* `json_ld_schema["author"] == "John"`
- **Array of Strings (Simple Tags):**
  - Use `JSON_CONTAINS`.
  - *Example:* `JSON_CONTAINS(json_ld_schema["tags"], "news")`
- **Combining Rules:**
  - *Correct:* `json_ld_schema["author"] == "John" and JSON_CONTAINS(json_ld_schema["tags"], "AI")`
- **Array of Objects (Complex Nested Data):**
  - **DO NOT FILTER.** This causes syntax errors.
  - *Input:* "Show me interview questions about Algorithms" (where "Algorithms" is inside an object in a list)
  - *Output:* NO_FILTER

### EXAMPLES
Input: articles written by admin
Output: json_ld_schema["author"] == "admin"

Input: show me PDF documents
Output: original_type == "pdf"

Input: content about security (security is a tag)
Output: JSON_CONTAINS(json_ld_schema["keywords"], "security")

Input: questions about Algorithms (Algorithms is inside a complex object list)
Output: NO_FILTER

### CURRENT REQUEST
Input: {user_query}
Output:"""



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


# Create a class for Milvus Knowledge Base Builder
class MilvusKnowledgeBaseBuilder:
    """Handles building knowledge base in Milvus vector database."""
    
    def __init__(self, domain: str, batch_id: str, is_openai: bool, query: str):
        self.domain = domain
        self.batch_id = batch_id
        self.is_openai = is_openai
        self.collection_name = f"collection_{clean_domain(self.domain)}_{clean_parent_batch_id(self.batch_id)}"
        self.query=query
        self.similarity_threshold = 0.3
        self.vector_client=None

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
    

    # Create Build for Milvus Knowledge Base
    def build(self):
        try:
            # Create Milvus Connection
            self._create_milvus_connection()
  
            # Get Response
            result=self._getResponse()

            return result
        
        except Exception as e:
            logger.error(f"Error building Milvus Knowledge Base: {e}")
            return "Error building Milvus Knowledge Base"

    # reqrite quary
    def _rewrite_query(self, user_query: str) -> str:
        try:
            llm = self._getLLM()

            # Get the optimized string
            formatted_prompt = REWRITE_QUERY_PROMPT.format(
                domain=self.domain, 
                user_query=user_query
            )
            
            response = llm.invoke(formatted_prompt)

            # Get Content only
            optimized_query = response.content
            
            print(f"🔄 Query Transformed: '{user_query}' -> '{optimized_query}'")
            return optimized_query.strip()
            
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}. Using original query.")
            return user_query
    
    def _get_collection_sample(self):
        """
        Fetches 1 record to understand the actual VALUES in the DB.
        """
        try:
            collection=self.vector_client.col

            res = collection.query(
                expr="", 
                limit=1, 
                output_fields=["source", "type", "domain", "original_type", "section","text","json_ld_schema"]
            )

            if not res:
                return None

            return json.dumps(res, default=str)    
        except Exception as e:
            logger.error(f"Error creating Milvus connection: {e}")

    def _generate_milvus_filter(self,schema_sample: str) -> str:
        """
        Asks LLM to write a Milvus boolean expression based on your specific Schema.
        """
        try:
            llm = self._getLLM()
            
            # We explicitly list YOUR fields in the prompt
            prompt = MILVUS_FILTER_PROMPT.format(
                user_query=self.query,
                schema_sample=schema_sample
            )
            
            response = llm.invoke(prompt)
            filter_expr = response.content if hasattr(response, 'content') else str(response)
            
            if "NO_FILTER" in filter_expr or len(filter_expr) < 2:
                print(f"Dynamic Filter: None (Broad Search)")
                return ""

            return filter_expr

        except Exception as e:
            logger.error(f"Filter generation failed: {e}")
            return ""
           

    # Get the Response
    def _getResponse(self):
        """
        Builds and executes the standard LangChain RAG pipeline:
        Retriever -> Document Chain -> LLM
        """
        try:
            llm = self._getLLM()

            # 1. Rewriting & Filtering (Pre-processing)
            search_query = self._rewrite_query(self.query)
            sample_schema = self._get_collection_sample()
            filter_expr = self._generate_milvus_filter(sample_schema)

            print(f"🔎 Final Search Query: {search_query}")
            print(f"🧬 Final Filter Expr: {filter_expr}")

            # 2. Create the Retriever (The "Tool" replacement)
            search_kwargs = {"k": 3}
            if filter_expr:
                search_kwargs["expr"] = filter_expr

            retriever = self.vector_client.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )

            # 3. Create the Prompt Template
            # This combines the System Persona with the retrieved Context
            prompt = ChatPromptTemplate.from_messages([
                ("system", LUFFY_RAG_PROMPT),
                ("human", "{input}"),
            ])

            # 4. Create the Document Chain (Combines retrieved docs into the prompt)
            question_answer_chain = create_stuff_documents_chain(llm, prompt)

            # 5. Create the Retrieval Chain (Orchestrates Retrieval + QA)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            # 6. Execute
            response = rag_chain.invoke({"input": search_query})

            print(f"Luffy's Answer: {response['answer']}")
            return response["answer"]

        except Exception as e:
            logger.error(f"Error in RAG Chain: {e}")
            return "Argh! Something went wrong retrieving the treasure map!"

    # Get LLM
    def _getLLM(self):
        try:
            if not os.environ.get("GROQ_API_KEY"): raise ValueError("GROQ_API_KEY missing.")
            return ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, groq_api_key=os.environ.get("GROQ_API_KEY"))
        except Exception as e:
            logger.error(f"Error getting LLM: {e}")
            return HuggingFaceEndpoint(
                huggingfacehub_api_token=os.getenv("HuggingFace_API_KEY"),
                repo_id="HuggingFaceH4/zephyr-7b-beta"
            )
        
    # Create milvus Connection
    def _create_milvus_connection(self):
        try:
            self.vector_client = Milvus(
                embedding_function=self.embedder,
                collection_name=self.collection_name,
                connection_args={
                    "host": os.getenv("HOST"),
                    "port": os.getenv("PORT"),
                },
            )
        except Exception as e:
            logger.error(f"Error creating Milvus connection: {e}")
    
   