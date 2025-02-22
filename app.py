# app.py
import os
import json
import logging
import uuid

from flask import Flask, request, jsonify, render_template, Response, stream_with_context, session
from flask_migrate import Migrate
from dotenv import load_dotenv

from langsmith import Client
from langchain_core.tracers.context import collect_runs
from langchain_core.tracers.langchain import wait_for_all_tracers

# Build absolute paths for flask
base_dir = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(base_dir, "static", "templates")
static_dir = os.path.join(base_dir, "static")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

# Load environment variables
load_dotenv()
company = os.getenv("EMPRESA", "Mi Empresa")
usuario = os.getenv("USUARIO")
secret_key = os.getenv("SECRET_KEY")

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = secret_key
app.logger.setLevel(logging.INFO)

# Optionally, also configure Flask's app.logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import your new streaming helper from assistant_chain_lcel
from chains.assistant_chain_lcel import run_chain_stream, post_process_fn
from utilities.rag_service import RAGService
from utilities.instruction_parser import InstructionParser

logger.info(f"static_folder = {app.static_folder}")
logger.info(f"static_url_path = {app.static_url_path}")

# Setup Langsmith
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = company
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"

# Initialize components:
rag_service = RAGService()
domain = rag_service.domain

# Instructions parser
instruction_parser = InstructionParser("instructions.json")



@app.route("/", methods=["GET"])
def home():
    """Serve the main HTML page."""
    return render_template("index.html", company=company, domain=domain)


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Receive run_id, score, and user comment.
    Send them to LangSmith via client.create_feedback().
    """
    data = request.get_json()
    run_id = data.get("run_id")
    score = data.get("score")
    comment = data.get("feedback", "")

    if not run_id:
        return jsonify({"message": "Missing run_id"}), 400

    client = Client()  # You can instantiate at the top-level if you prefer

    try:
        client.create_feedback(
            run_id=run_id,
            key="ui-feedback",
            score=score,
            comment=comment
        )
        return jsonify({"message": "¡Gracias por tu evaluación en LangSmith!"}), 200
    except Exception as e:
        logger.error(f"Error sending feedback to LangSmith: {e}")
        return jsonify({"message": "Error interno al enviar feedback a LangSmith."}), 500


@app.route("/check_rag", methods=["POST"])
def check_rag():
    """
    Check if the query should trigger a GroundX (RAG) search.
    """
    data = request.get_json()
    user_message = data.get("message", "")
    rag_used = rag_service.should_call_groundx(user_message)
    # Store the result in session
    session["rag_decision"] = rag_used
    session["cached_message"] = user_message
    return jsonify({"is_rag": rag_used})

#NEW THUMB FEEDBACK
@app.route("/thumb_feedback", methods=["POST"])
def thumb_feedback():
    """
    Receives {run_id, evaluation="up"|"down", reason=""}
    Sends a feedback to LangSmith with "Thumb score" key,
    up=1.0, down=0.0, comment with reason if any.
    """
    data = request.get_json(force=True) or {}

    run_id = data.get("run_id", "").strip()
    evaluation = data.get("evaluation", "").strip().lower()
    reason = data.get("reason", "").strip()

    if not run_id:
        return jsonify({"error": "No run_id provided"}), 400

    if evaluation not in ["up", "down"]:
        return jsonify({"error": "evaluation must be 'up' or 'down'"}), 400

    # Convert up/down to numeric
    score = 1.0 if evaluation == "up" else 0.0

    client = Client()
    try:
        client.create_feedback(
            run_id=run_id,
            key="Thumb score",
            score=score,
            comment=f"User thumb feedback: {reason}" if reason else "User thumb feedback",
        )
        return jsonify({"message": "Feedback registered in LangSmith."}), 200

    except Exception as e:
        logging.exception("Error sending thumb feedback to LangSmith.")
        return jsonify({"error": str(e)}), 500

@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    """
    Stream the chat completion response chunk-by-chunk using the new LCEL-based chain.
    """
    data = request.get_json()
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"message": "Error: No message provided"}), 400

    # Load your system prompt from instructions.json as needed
    system_prompt = instruction_parser.load_instruction()

    # Build the chain input
    chain_input = {
        "query": user_message,
        "system_prompt": system_prompt,
        "company": os.getenv("EMPRESA"),
        "domain": rag_service.domain,

     }

    # Check if we have a cached RAG classification for this same user_message
    if session.get("cached_message") == user_message:
        # Reuse the same rag decision
        chain_input["already_classified"] = True
        chain_input["rag_decision"] = session.get("rag_decision", False)
    else:
        # fallback: no match or no classification stored
        chain_input["already_classified"] = False

    logger.info(chain_input['query'])

    # Generate a unique run_id & session_id for each user message (or manage your own session logic)
    run_id = str(uuid.uuid4())
    session_id = usuario

    def generate():
        with collect_runs() as cb:
            streamed_chunks = []
            last_chunk = None

            try:
                # Stream from the new LCEL-based chain
                for partial in run_chain_stream(chain_input, run_id, session_id):
                    # partial might be an AIMessageChunk (which has .content)
                    # or a plain string. Let's handle both:
                    if hasattr(partial, "content"):
                        chunk_text = partial.content
                        # If usage info is attached, keep track
                        last_chunk = partial
                    else:
                        chunk_text = str(partial)

                    streamed_chunks.append(chunk_text)
                    yield chunk_text  # yield partial chunk to the client in real-time

                # Once done streaming, do final references/number localization
                entire_llm_output = "".join(streamed_chunks)
                final_answer_localized = post_process_fn(entire_llm_output)
                yield "\n[REF_POSTPROCESS]" + final_answer_localized


                # If you want to log usage from the last chunk
                if last_chunk and hasattr(last_chunk, "usage_metadata"):
                    usage_data = last_chunk.usage_metadata
                    logger.info(f"Total input tokens used: {usage_data.get('input_tokens')}")
                    logger.info(f"Total output tokens used: {usage_data.get('output_tokens')}")

                retrieval_run_id_stream = cb.traced_runs[0].id.urn
                score = rag_service.total_score
                client = Client()
                client.create_feedback(
                    retrieval_run_id_stream,
                    key="Total-RAG-score",
                    score=score,
                    source_run_id=retrieval_run_id_stream,
                    comment="Puntuacion del RAG obtenido con la consulta (stream).",
                )

                yield f"\n[RUN_ID]{retrieval_run_id_stream}"

            except Exception as e:
                logger.error(f"Error in /chat_stream: {e}")
                yield f"\n[ERROR] {str(e)}"
            finally:
                # Ensure we flush tracer buffers
                wait_for_all_tracers()

    # Return a streaming response
    return Response(stream_with_context(generate()), mimetype='text/plain')


@app.route("/process_references", methods=["POST"])
def process_references():
    """
    Process text references and return the text with citations replaced by links.
    """
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        logger.warning("No text provided for reference processing.")
        return jsonify({"message": "Error: No se proporcionó texto."}), 400

    try:
        processed_text = rag_service.process_references_in_text(text)
        logger.info("References processed successfully.")
        return jsonify({"processed_text": processed_text}), 200
    except Exception as e:
        logger.error(f"Error processing references: {e}")
        return jsonify({"message": "Error al procesar referencias."}), 500

if __name__ == "__main__":
    app.run(debug=False, port=5000)
