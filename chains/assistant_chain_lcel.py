import os
import logging
import uuid
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.globals import set_debug
from langchain_core.tracers.context import collect_runs
from langchain_core.tracers.langchain import wait_for_all_tracers

from langsmith import Client
from langsmith.run_trees import RunTree

# For memory and summarization
from utilities.memory_utils import get_session_history, summarizer_lcel

# Import your existing services and utilities.
from utilities.rag_service import RAGService
from utilities.reference_maker import ReferenceMaker
from utilities.instruction_parser import InstructionParser

# ------------------------------------------------------------------
# Logging and environment setup
# ------------------------------------------------------------------
logger = logging.getLogger(__name__)
load_dotenv()

# Assume rag_service is already initialized (e.g., in your app code)
rag_service = RAGService()

# Setup Langsmith
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Urufarma"
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"

# ------------------------------------------------------------------
# 1) Define the new “Parallel RAG” approach
# ------------------------------------------------------------------

# A) Spanish-Only Search
def spanish_search_fn(inputs: dict) -> dict:
    """
    Receives {"query": "some Spanish text"}
    Returns {"spanish_results": [GroundX results...]}
    """
    newdict = dict(inputs)
    query_spanish = inputs["query"]
    results_es = rag_service.groundx_search_spanish_only(query_spanish)
    # add the results to the dictionary
    newdict["spanish_results"] = results_es
    newdict["system_prompt"] = newdict['system_prompt']
    return newdict

spanish_search_runnable = RunnableLambda(spanish_search_fn)

# B) Translator -> English Search sub-chain
def translator_fn(inputs: dict) -> dict:
    """
    Takes {"query": "Spanish text"}
    Returns {"english_query": "English text"}
    """
    newdict = dict(inputs)  # keep domain, company, query, etc.
    span_query = inputs["query"]
    eng_query = rag_service.translate_spanish_to_english(span_query)
    newdict["english_query"] = eng_query
    return newdict
translator_runnable = RunnableLambda(translator_fn)

def english_search_fn(inputs: dict) -> dict:
    """
    Takes {"english_query": "..."}
    Returns {"english_results": [GroundX results...]}
    """
    newdict = dict(inputs)
    query_eng = inputs["english_query"]
    results_en = rag_service.groundx_search_english_only(query_eng)
    newdict["english_results"] = results_en
    return newdict

english_search_runnable = RunnableLambda(english_search_fn)

# Combine them into a pipeline: translator -> englishSearch
english_search_chain = translator_runnable | english_search_runnable

# C) RunnableParallel: Spanish + (translator->English) in parallel
rag_parallel = RunnableParallel(
    spanish=spanish_search_runnable,
    english=english_search_chain
)

# D) Merge Step
def merge_rag_fn(inputs: dict) -> dict:
    """
    inputs: {
      "spanish": {"spanish_results": [...], ...any other keys...},
      "english": {"english_results": [...], ...any other keys...}
      "query": "...original query..."  (since parallel passes it along)
      ...
    }
    We'll combine them into a single text. Store in inputs["context"].
    Also set rag_service.total_score if desired.
    """

    # 1) Start from the 'spanish' dict
    out = dict(inputs["spanish"])  # i.e. copy all keys from 'spanish' branch

    # 2) Overlay keys from 'english'
    english_dict = inputs.get("english", {})
    for k, v in english_dict.items():
        out[k] = v

    # 3) Now do your combined Spanish+English result logic
    spanish_results = out.get("spanish_results", [])
    english_results = out.get("english_results", [])


    # 2) filter, score, sort, build text
    combined_list = []
    span_score = 0
    eng_score = 0

    for res in spanish_results:
        score = res.score
        if score >= 150:
            span_score += score
            combined_list.append(("ES", score, res))

    for res in english_results:
        score = res.score
        if score >= 150:
            eng_score += score
            combined_list.append(("EN", score, res))

    combined_list.sort(key=lambda x: x[1], reverse=True)

    # Build final text
    chunks = []
    for (lang, scr, doc) in combined_list:
        file_name = getattr(doc, "file_name", f"Documento {lang}")
        snippet = (
            "----------\n"
            f"The following text excerpt is from a document named {file_name}:\n\n"
            f"{doc.suggested_text}\n"
        )
        chunks.append(snippet)

    if chunks:
        final_context = "\n".join(chunks).strip()
    else:
        final_context = (
            "No documents retrieved for this question. "
            "Respond using only your general knowledge."
        )

    # Set the final context
    out["context"] = final_context

    # record total score if you want
    rag_service.total_score = span_score + eng_score
    logger.info(f"Merged Spanish Score={span_score}, English Score={eng_score}, total={rag_service.total_score}")

    return out

merge_rag_runnable = RunnableLambda(merge_rag_fn)

# E) Combine parallel + merge into one sub-chain
rag_parallel_merge = rag_parallel | merge_rag_runnable

# F) Condition: if should_call_groundx is false, skip parallel
def conditional_rag_fn(inputs: dict) -> dict:
    query = inputs["query"]
    if rag_service.should_call_groundx(query):
        # parallel approach
        return rag_parallel_merge.invoke(inputs)
    else:
        # fallback
        rag_service.total_score = 0
        inputs["context"] = (
            "No documents retrieved for this question. "
            "Respond using only your general knowledge."
        )
        return inputs

conditional_rag_retriever = RunnableLambda(conditional_rag_fn)


# ------------------------------------------------------------------
# 2) Summarizer Step (from memory_utils.py)
# ------------------------------------------------------------------
from utilities.memory_utils import summarizer_lcel

# ------------------------------------------------------------------
# 3) Final Prompt Generator
# ------------------------------------------------------------------
def final_prompt_generator_fn(inputs: dict):
    """
    Expects inputs to contain:
      - "system_prompt": an uninvoked ChatPromptTemplate from InstructionParser
      - "company": the client/company name
      - "domain": the domain
      - "context": the retrieved factual context
      - "query": the user query
    """
    company = inputs['company']
    domain = inputs['domain']

    filled_system_prompt = inputs['system_prompt'].invoke({"company": company, "domain": domain})
    base_system_message = filled_system_prompt.messages[0].content

    conversation_string = ""
    if "history" in inputs:
        for msg in inputs["history"]:
            role = "User" if msg.type == "human" else "Assistant"
            conversation_string += f"{role}: {msg.content}\n"

    final_system_message = (
        base_system_message
        + "\n===\nHISTORIAL DE LA CONVERSACIÓN\n"
        + conversation_string
        + "\n===\nFACTUAL CONTEXT:\n"
        + inputs["context"]
        + "\n===\n"
        + "Note: Use the context above to answer."
    )

    logger.info(final_system_message)
    messages = [
        SystemMessage(content=final_system_message),
        HumanMessage(content=inputs['query'])
    ]
    return messages

final_prompt_generator = RunnableLambda(final_prompt_generator_fn)

# ------------------------------------------------------------------
# 4) The LLM call (streaming chat model)
# ------------------------------------------------------------------
from langchain_openai import ChatOpenAI
chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, streaming=True, stream_usage=True)

# ------------------------------------------------------------------
# 5) Post-processor
# ------------------------------------------------------------------
from utilities.reference_maker import ReferenceMaker

def post_process_fn(answer: str) -> str:
    processed = rag_service.process_references_in_text(answer)
    localized = ReferenceMaker.convert_us_to_local(processed, src_locale='en_US', target_locale='es_ES')
    return localized

# ------------------------------------------------------------------
# 6) Compose the LCEL pipeline
#    Step by Step:
#    1) (conditional_rag_retriever => parallel if relevant)
#    2) summarizer_lcel
#    3) final_prompt_generator
#    4) LLM
# ------------------------------------------------------------------
assistant_chain_lcel_core = (
    conditional_rag_retriever
    | summarizer_lcel
    | final_prompt_generator
    | chat_model
)

# ------------------------------------------------------------------
# 7) Wrap the pipeline with memory
# ------------------------------------------------------------------
from utilities.memory_utils import get_session_history

assistant_chain_lcel = RunnableWithMessageHistory(
    runnable=assistant_chain_lcel_core,
    get_session_history=get_session_history,
    input_messages_key="query",
    history_messages_key="history"
)

# ------------------------------------------------------------------
# 8) Provide a helper function for synchronous streaming
# ------------------------------------------------------------------
def run_chain_stream(chain_input: dict, run_id, session_id="default_session"):
    if run_id is None:
        run_id = str(uuid.uuid4())

    config = {
        "run_id": run_id,
        "configurable": {"session_id": session_id},
        "metadata": {"user_query": chain_input.get("query", "<no query>")},
        "tags": ["Urufarma asistente"]
    }
    return assistant_chain_lcel.stream(chain_input, config=config)

# ------------------------------------------------------------------
# 9) Demo usage
# ------------------------------------------------------------------
# if __name__ == "__main__":
#     set_debug(False)
#     load_dotenv()
#
#     from utilities.instruction_parser import InstructionParser
#     from langsmith import Client
#
#     company = os.getenv("EMPRESA")
#     domain = os.getenv("DOMINIO")
#     usuario = os.getenv("USUARIO")
#
#     system_prompt = InstructionParser("../instructions.json").load_instruction()
#
#     run_id = str(uuid.uuid4())
#     session_id = usuario
#
#     chain_input = {
#         "query": "Hola, ¿Me podes describir el plan de mantenimiento de la bomba de calor de P6?",
#         "system_prompt": system_prompt,
#         "company": company,
#         "domain": domain,
#     }
#
#     stream_run_id = str(uuid.uuid4())
#     with collect_runs() as cb:
#         try:
#             streamed_chunks = []
#             last_chunk = None
#             for partial in run_chain_stream(chain_input, stream_run_id, session_id):
#                 if hasattr(partial, "content"):
#                     chunk_text = partial.content
#                 else:
#                     chunk_text = partial
#
#                 streamed_chunks.append(chunk_text)
#                 print(chunk_text, end="", flush=True)
#                 last_chunk = partial
#
#             print("\n-- Done streaming! --")
#
#             if last_chunk and hasattr(last_chunk, "usage_metadata"):
#                 usage_data = last_chunk.usage_metadata
#                 logger.info(f"Total input tokens used: {usage_data.get('input_tokens')}")
#                 logger.info(f"Total output tokens used: {usage_data.get('output_tokens')}")
#
#             entire_llm_output = "".join(streamed_chunks)
#             final_processed = post_process_fn(entire_llm_output)
#             print("\n-- Final Post-Processed Answer --")
#             print(final_processed)
#
#             retrieval_run_id_stream = cb.traced_runs[0].id.urn
#             client = Client()
#             client.create_feedback(
#                 retrieval_run_id_stream,
#                 key="Total-RAG-score",
#                 score=rag_service.total_score,
#                 source_run_id=stream_run_id,
#                 comment="Score from parallel RAG search.",
#             )
#         finally:
#             wait_for_all_tracers()
