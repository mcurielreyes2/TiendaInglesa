# utilities/rag_service.py
import os
import json
import time
import logging
import openai
from groundx import GroundX
from utilities.reference_maker import ReferenceMaker
from langsmith import Client

logger = logging.getLogger(__name__)

class RAGService:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groundx_api_key = os.getenv("GROUNDX_API_KEY")
        self.bucket_id = os.getenv("GROUNDX_BUCKET_ID")

        if not self.openai_api_key or not self.groundx_api_key or not self.bucket_id:
            try:
                with open("config.json") as config_file:
                    config = json.load(config_file)
                    self.openai_api_key = self.openai_api_key or config.get("OPENAI_API_KEY")
                    self.groundx_api_key = self.groundx_api_key or config.get("GROUNDX_API_KEY")
                    self.bucket_id = self.bucket_id or config.get("GROUNDX_BUCKET_ID")
            except FileNotFoundError:
                raise ValueError("No API key or bucket ID found in environment variables or config.json.")

        if not self.bucket_id or not str(self.bucket_id).isdigit():
            raise ValueError("GROUNDX_BUCKET_ID must be a valid integer.")
        self.bucket_id = int(self.bucket_id)

        # Initialize GroundX and OpenAI clients
        self.groundx = GroundX(api_key=self.groundx_api_key)
        self.client = openai
        self.total_score = 0

        # Load any keywords
        self.coffee_keywords = self.load_coffee_keywords("kw.txt")

        docs_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static", "docs")
        self.reference_maker = ReferenceMaker(docs_directory=docs_directory, threshold=70)

    def load_coffee_keywords(self, filename: str):
        project_root = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(project_root, "..")
        file_path = os.path.join(project_root, filename)
        keywords = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    keywords.append(line.lower())
        except FileNotFoundError:
            logger.warning(f"Could not find {filename}, defaulting to empty keyword list.")
        return keywords

    def should_call_groundx(self, query: str) -> bool:
        lower_query = query.lower()
        for kw in self.coffee_keywords:
            if kw in lower_query:
                logger.info(f"Found keyword '{kw}' => definitely about Urufarma.")
                return True

        classification_prompt = f"""
        Eres un clasificador de textos. Tu te encargas de definir si la consulta realizada por el usuario es relevante para el contexto para el que este programa fue diseñado. 
        Dada la consulta del usuario, estima la probabilidad (0-100) de que la consulta sea sobre HVAC, eficiencia energética, la empresa Urufarma o una temática relacionada.
        Devuelve SOLO un número del 0 al 100 (un entero). Sin texto adicional.
        User query: {query}
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": classification_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0
        )
        result_text = response.choices[0].message.content.strip()
        try:
            probability = float(result_text)
        except ValueError:
            logger.info(f"Unexpected classification response: '{result_text}'. Defaulting to 50.")
            probability = 50.0

        threshold = 50
        logger.info(f"Urufarma probability: {probability}% (threshold={threshold})")
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
