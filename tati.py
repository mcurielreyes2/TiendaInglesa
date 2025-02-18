import streamlit as st
import asyncio
import httpx
import json
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers import LangChainTracer  # New tracer import
from langsmith import traceable  # Traceable decorator for GroundX fetch

# Load environment variables from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROUNDX_API_KEY = os.getenv("GROUNDX_API_KEY")
GROUNDX_BUCKET_ID = os.getenv("GROUNDX_BUCKET_ID")

# Set the environment variables required for LangSmith tracing:
# export LANGSMITH_TRACING=true
# export LANGSMITH_API_KEY="<your-langsmith-api-key>"

# Initialize the LangSmith tracer and wrap it in a callback manager
tracer = LangChainTracer()
callback_manager = CallbackManager([tracer])

def remove_code_fences(text):
    """
    Removes markdown code fences (```json ... ```) from the given text.
    """
    if not isinstance(text, str):
        return text  # Return as is if not a string
    if text.startswith("```json"):
        lines = text.splitlines()
        if lines[0].startswith("```json"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text

def fix_question(question):
    """
    Fixes grammatical and punctuation errors in the question using LangChain.
    """
    fix_template = (
        "Corrige los errores gramaticales y de puntuación en la siguiente pregunta:\n\n"
        "{question}\n\n"
        "Devuelve solo el texto corregido."
    )
    formatted_prompt = fix_template.format(question=question)
    llm = ChatOpenAI(
        temperature=0.0,
        model_name="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        callbacks=callback_manager  # Use the shared callback manager
    )
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    return response.content.strip()

def generate_standalone_questions(question):
    """
    Generates 3 self-contained questions from the original question using LangChain.
    """
    gen_template = (
        "Dada la siguiente pregunta, genera 3 preguntas autónomas y auto-contenidas "
        "que reformulen la consulta original desde diferentes perspectivas. "
        "Devuelve estrictamente un JSON con el siguiente formato:\n\n"
        '{{"questions": ["Pregunta 1", "Pregunta 2", "Pregunta 3"]}}\n\n'
        "Pregunta original: {question}"
    )
    formatted_prompt = gen_template.format(question=question)
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        callbacks=callback_manager
    )
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    output = response.content

    # Parse the output as JSON, after removing code fences if present
    if isinstance(output, dict):
        data = output
    else:
        output = remove_code_fences(output)
        try:
            data = json.loads(output)
        except Exception as e:
            st.error(f"Error al analizar el JSON de la salida de OpenAI: {e}\nSalida recibida: {output}")
            return []
    questions = data.get("questions", [])
    if len(questions) != 3:
        st.error("Se esperaban 3 preguntas en la salida JSON, pero se obtuvo un número diferente.")
        return []
    return questions

def generate_final_answer(question, context):
    """
    Generates the final answer by combining the original question and the retrieved context.
    """
    final_template = (
        "Eres un asistente útil. Basándote en la pregunta original y en el siguiente contexto recuperado, "
        "proporciona una respuesta detallada y útil.\n\n"
        "Pregunta Original: {question}\n\n"
        "Contexto Recuperado:\n{context}\n\n"
        "Respuesta:"
    )
    formatted_prompt = final_template.format(question=question, context=context)
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY,
        callbacks=callback_manager
    )
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    return response.content.strip()

@traceable
async def fetch_groundx(question, bucket_id, api_key):
    """
    Fetches context from the GroundX API for the given question.
    """
    url = f"https://api.groundx.ai/api/v1/search/{bucket_id}"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    payload = {"query": question, "n": 2}
    print(question)
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get("search", "").get("text", "")

async def fetch_all_groundx(questions, bucket_id, api_key):
    """
    Performs concurrent GroundX API calls for each question.
    """
    tasks = [fetch_groundx(q, bucket_id, api_key) for q in questions]
    return await asyncio.gather(*tasks, return_exceptions=True)

@traceable
def solve_question_chain(question):
    corrected_question = fix_question(question)

    standalone_questions = generate_standalone_questions(corrected_question)
    if not standalone_questions:
        st.error("No se pudieron generar preguntas autónomas.")
        return

    all_questions = [corrected_question] + standalone_questions

    results = asyncio.run(fetch_all_groundx(all_questions, GROUNDX_BUCKET_ID, GROUNDX_API_KEY))

    context = f"**Pregunta:** {corrected_question}\n**Respuesta:** {results[0]}\n\n"
    for q, r in zip(standalone_questions, results[1:]):
        context += f"**Pregunta:** {q}\n**Respuesta:** {r}\n\n"

    final_answer = generate_final_answer(corrected_question, context)

    return final_answer

def main():
    st.title("Aplicación de Chat Bot utilizando LangChain, OpenAI y GroundX")

    if not OPENAI_API_KEY or not GROUNDX_API_KEY or not GROUNDX_BUCKET_ID:
        st.error(
            "No se encontraron las claves API y/o el ID del Bucket en el archivo .env. "
            "Por favor, crea un archivo .env con las claves necesarias."
        )
        return

    user_question = st.text_area("Introduce tu pregunta")

    if st.button("Enviar"):
        if not user_question:
            st.error("Por favor, introduce una pregunta.")
            return

        with st.spinner("Answering la pregunta..."):
            final_answer = solve_question_chain(user_question)
        st.write("Respuesta", final_answer)


if __name__ == "__main__":
    main()
