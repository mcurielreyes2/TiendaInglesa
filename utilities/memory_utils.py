# # utilities/memory_utils.py
import os
import logging
from dotenv import load_dotenv
from typing import List, Dict
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# 1. Retrieve the API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("No OPENAI_API_KEY found in environment.")

# 2. Use it when creating the summarizer LLM
summary_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.0,
    max_tokens=256,
    openai_api_key=OPENAI_API_KEY,
)


# ------------------------------------------------------------------
# 1. In-Memory Chat History
# ------------------------------------------------------------------

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """Simple in-memory storage of chat messages."""
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Append new messages to the conversation."""
        self.messages.extend(messages)

    def clear(self) -> None:
        """Clear the entire conversation history."""
        self.messages = []


# A global dictionary to store session_id -> InMemoryHistory.
# In a real app, you might replace this with Redis or a DB.
session_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return the conversation history object for a given session."""
    if session_id not in session_store:
        session_store[session_id] = InMemoryHistory()
    return session_store[session_id]


# ------------------------------------------------------------------
# 2. Summarizer Step
#    - If 'history' has more than 4 messages, call a small LLM to
#      produce a concise summary. Replace 'history' with that summary.
# ------------------------------------------------------------------

SUMMARY_THRESHOLD = 8

# A small LLM just for summarizing (use whichever model you like)
summary_llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.0,
    max_tokens=256
)

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Eres un asistente que resume sucintamente la conversación hasta ahora."
        "Concéntrate en los puntos esenciales; mantenlo coherente y breve."
        "Solo muestra el texto del resumen, sin comentarios adicionales."
    )),
    ("user", "{transcript}")
])

def _summarize_history_fn(inputs: Dict) -> Dict:
    """
    If inputs["history"] has more than SUMMARY_THRESHOLD messages,
    1) Summarize them into a single message,
    2) Replace the entire 'history' list with that single summary.
    """
    if "history" not in inputs:
        return inputs  # no conversation in input

    history = inputs["history"]
    if len(history) <= SUMMARY_THRESHOLD:
        return inputs  # still small enough, no summarization

    # Build a text transcript from the entire conversation
    lines = []
    for msg in history:
        role = "Human" if msg.type == "human" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    transcript = "\n".join(lines)

    # Invoke the summarizer prompt
    prompt_value = summary_prompt.invoke({"transcript": transcript})
    summary_result = summary_llm.invoke(prompt_value.messages)

    # Replace the full 'history' with a single summary message
    inputs["history"] = [AIMessage(content=summary_result.content)]

    return inputs

summarizer_lcel = RunnableLambda(_summarize_history_fn)
