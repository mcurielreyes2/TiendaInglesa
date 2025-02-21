# utilities/rag_service.py
import os
import json
import time
import logging
import openai
from groundx import GroundX
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from utilities.reference_maker import ReferenceMaker

logger = logging.getLogger(__name__)

class RAGService:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Singleton check
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

        # 1) Read environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groundx_api_key = os.getenv("GROUNDX_API_KEY")
        self.empresa = os.getenv("EMPRESA", "default_company")

        # 2) Load your brand-specific config.json
        self.config = self.load_config_json(self.empresa)

        # 3) Buckets & domain
        self.buckets = self.config.get("buckets", [])
        self.bucket_id = self.select_bucket_for_query("default_init")  # picks a default from self.buckets
        self.domain = self.config.get("domain", "domain_not_found")
        self.keywords = self.config.get("keywords", [])

        # 4) Validate environment keys + chosen bucket
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing in environment variables.")
        if not self.groundx_api_key:
            raise ValueError("GROUNDX_API_KEY is missing in environment variables.")
        if not self.bucket_id or not str(self.bucket_id).isdigit():
            raise ValueError("GROUNDX_BUCKET_ID must be a valid integer. (Check your config's 'buckets' array.)")

        # 5) Initialize your chat model
        self.chat_model = ChatOpenAI(model="gpt-4o-mini",temperature=0.0)

        # 6) Initialize GroundX client & references
        self.groundx = GroundX(api_key=self.groundx_api_key)
        self.client = openai
        self.total_score = 0

        docs_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","static",self.empresa,"docs" )
        self.reference_maker = ReferenceMaker(docs_directory=docs_directory, threshold=70)

        logger.info(f"RAGService initialized for company={self.empresa}, domain={self.domain}, bucket_id={self.bucket_id}")

    def load_config_json(self, company):
        file_path = os.path.join("config", company, "config.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data


    def should_call_groundx(self, query: str) -> bool:

        lower_query = query.lower()
        for kw in self.keywords:
            if kw in lower_query:
                logger.info(f"Found keyword '{kw}' => definitely about {self.empresa} or/and {self.domain}.")
                return True

        classification_prompt = ChatPromptTemplate([
            ("system",
             """Eres un clasificador de textos. Tu labor es definir si la consulta realizada 
                por el usuario es relevante para el contexto para el que este programa fue diseñado. 
                Dada la consulta del usuario, estima la probabilidad (0-100) de que la consulta sea 
                sobre {domain}, la empresa {company} o una temática relacionada.
                Devuelve SOLO un número del 0 al 100 (un entero). Sin texto adicional."""),
            ("user", "User query: {query}")
        ])

        filled_prompt = classification_prompt.invoke({
            "domain": self.domain,
            "company": self.empresa,
            "query": query
        })

        #Call the model via LangChain
        response = self.chat_model.invoke(filled_prompt)
        result_text = response.content.strip()

        try:
            probability = float(result_text)
        except ValueError:
            logger.info(f"Unexpected classification response: '{result_text}'. Defaulting to 50.")
            probability = 50.0

        threshold = 50
        logger.info(f"{self.empresa} and/or {self.domain} probability: {probability}% (threshold={threshold})")
        return probability >= threshold

    # ------------------------------------------------------------------
    # Single-language search methods
    # ------------------------------------------------------------------
    def groundx_search_spanish_only(self, query_spanish: str):
        """
        Return the raw GroundX results (a list of search results) for Spanish query only.
        We'll let the chain handle merging + scoring.
        """
        t0 = time.time()
        content_response_es = self.groundx.search.content(
            id=self.bucket_id,
            n=5,
            query=query_spanish
        )
        dt = time.time() - t0
        logger.info(f"Spanish search took {dt:.3f}s for query='{query_spanish}'")
        return content_response_es.search.results

    def groundx_search_english_only(self, query_english: str):
        """
        Return the raw GroundX results for an English query only.
        We'll do merges externally.
        """
        t0 = time.time()
        content_response_en = self.groundx.search.content(
            id=self.bucket_id,
            n=5,
            query=query_english
        )
        dt = time.time() - t0
        logger.info(f"English search took {dt:.3f}s for query='{query_english}'")
        return content_response_en.search.results

    # ------------------------------------------------------------------
    # Bucket ID selection
    # ------------------------------------------------------------------

    def select_bucket_for_query(self, query: str) -> str:
        """
        For now, we only have one bucket, so we just return that.
        In the future, you can add logic to pick from self.buckets
        based on the user's query or domain classification.

        """
        if not self.buckets:
            logger.warning("No buckets defined in config. Defaulting to 'BUCKET_ID_MISSING'")
            return "BUCKET_ID_MISSING"

        # If you only have one bucket:
        if len(self.buckets) == 1:
            chosen = self.buckets[0]
            logger.info(f"Only one bucket available. Using bucket_id={chosen['bucket_id']}")
            return chosen["bucket_id"]

        # Fallback if multiple are found but no logic implemented:
        default_choice = self.buckets[0]
        logger.info(f"Multiple buckets present. Defaulting to first: {default_choice['bucket_id']}")
        return default_choice["bucket_id"]



    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------
    def translate_spanish_to_english(self, text: str) -> str:
        translation_prompt = f"""
            Translate the following text from Spanish to English.
            Output only the translated text, nothing else.
            Text to translate:
            {text}
        """
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a translator. You translate Spanish text into English."},
                {"role": "user", "content": translation_prompt}
            ],
            temperature=0,
            max_tokens=1000
        )
        english_translation = response.choices[0].message.content.strip()
        return english_translation

    # ------------------------------------------------------------------
    # For reference processing in final answers
    # ------------------------------------------------------------------
    def process_references_in_text(self, text: str) -> str:
        """
        Replaces references in the text with actual links or citations, etc.
        """
        logger.info("Processing references in text via ReferenceMaker.")
        processed_text = self.reference_maker.process_text_references_with_citations(text)
        logger.info("References processed.")
        return processed_text
