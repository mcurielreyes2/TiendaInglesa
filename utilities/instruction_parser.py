# utilities/instruction_parser.py
import json
import logging
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class InstructionParser:
    def __init__(self, filepath: str):
        """
        Initialize the InstructionParser with the path to the JSON file.
        """
        self.filepath = filepath

    def load_instruction(self) -> ChatPromptTemplate:
        """
        Load and format the instruction text from a JSON file.
        """
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            instr = data["instruction"]

            general = instr["general"]
            documents = instr["document_summaries"]

            response_guidelines = "\n".join(instr["response_guidelines"])
            prioritization = instr["prioritization"]
            examples = "\n".join(instr["examples"])
            fallback = instr["fallback"]

            # Build the complete system prompt template text.
            system_template = (
                f"{general}\n\n"
                f"Response Guidelines:\n{response_guidelines}\n\n"
                f"Prioritization:\n{prioritization}\n\n"
                f"Examples:\n{examples}\n\n"
                f"Fallback:\n{fallback}\n"
                f"Documents:\n{documents}\n"
            )

            # Create a ChatPromptTemplate (which produces a ChatPromptValue with a system message)
            system_prompt = ChatPromptTemplate.from_messages([("system", system_template)])

            return system_prompt

        except Exception as e:
            logger.error(f"Error loading instruction: {e}")
            raise