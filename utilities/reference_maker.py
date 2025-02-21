# utilities/reference_maker.py
import os
import logging
from rapidfuzz import process, fuzz
from urllib.parse import quote
import re
from babel.numbers import parse_decimal, format_decimal

logger = logging.getLogger(__name__)

class ReferenceMaker:
    def __init__(self, docs_directory: str, threshold: int = 80):
        """
        Initialize the ReferenceMaker.
        """
        self.empresa = os.getenv("EMPRESA")
        self.docs_directory = docs_directory
        self.threshold = threshold

        if not os.path.exists(self.docs_directory):
            raise ValueError(f"El directorio de documentos no existe: {self.docs_directory}")

        self.docs_list = self.load_documents()

    def load_documents(self):
        try:
            files = os.listdir(self.docs_directory)
            files = [f for f in files if os.path.isfile(os.path.join(self.docs_directory, f))]
            logger.info(f"Documents loaded: {files}")
            return files
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return []

    def find_closest_filename(self, reference_name: str) -> str:
        normalized_ref = self.normalize_reference_name(reference_name)
        logger.info(f"Processing reference: {reference_name} (normalized: {normalized_ref})")
        match, score, _ = process.extractOne(
            normalized_ref,
            self.docs_list,
            scorer=fuzz.ratio
        )
        if score >= self.threshold:
            logger.info(f"Closest match found: {match} (score: {score}%)")
            return match
        else:
            logger.warning(f"No sufficiently similar match for '{reference_name}'. Best match: '{match}' with {score}% similarity.")
            return None

    def generate_document_link(self, exact_filename: str) -> str:
        encoded_filename = self.encode_filename_for_url(exact_filename)
        link = f"/static/docs/{encoded_filename}"
        logger.info(f"Generated link: {link}")
        return link

    def process_text_references_with_citations(self, text: str) -> str:
        doc_regex = re.compile(r'\*\*([^*]+)\*\*')
        matches = doc_regex.findall(text)
        if not matches:
            return text

        ref_map = {}
        ref_details = {}
        current_index = 1

        for ref_str in matches:
            if ref_str in ref_map:
                continue

            matched_filename = self.find_closest_filename(ref_str)
            if matched_filename:
                ref_map[ref_str] = current_index
                ref_details[current_index] = {
                    "ref_str": ref_str,
                    "matched_filename": matched_filename
                }
                current_index += 1

        def replacer(match_obj):
            ref_str_found = match_obj.group(1)
            if ref_str_found in ref_map:
                i = ref_map[ref_str_found]
                return f"**{ref_str_found}** <span class=\"doc-citation-number\">[{i}]</span>"
            else:
                return match_obj.group(0)

        text = doc_regex.sub(replacer, text)

        if ref_details:
            references_block = "\n\n<b>Referencias:</b>\n"
            for i in sorted(ref_details.keys()):
                info = ref_details[i]
                matched = info["matched_filename"]
                if matched:
                    link = f"static/{self.empresa}/docs/{self.encode_filename_for_url(matched)}"
                    references_block += f"<li>[{i}] <a href=\"{link}\" target=\"_blank\">{matched}</a></li>"
            text += references_block

        return text

    @staticmethod
    def convert_us_to_local(text, src_locale='en_US', target_locale='es_ES'):
        # If the input is not a string but has a "content" attribute (e.g. an AIMessage),
        # use that attribute.
        pattern = re.compile(r"\b\d{1,3}(,\d{3})*(\.\d+)?\b")

        def replacer(match):
            us_number_str = match.group(0)
            try:
                parsed = parse_decimal(us_number_str, locale=src_locale)
                return format_decimal(parsed, locale=target_locale)
            except Exception:
                return us_number_str

        return pattern.sub(replacer, text)

    @staticmethod
    def normalize_reference_name(reference_name: str) -> str:
        return reference_name.replace("+", " ").replace("%20", " ").replace("%28", "(").replace("%29", ")")

    @staticmethod
    def encode_filename_for_url(filename: str) -> str:
        return quote(filename)
