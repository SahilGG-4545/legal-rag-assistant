import os
import tempfile
import threading
from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv
from rag_index_builder import build_index_from_pdf
from tools import retrieve_legal_context
from autogen import AssistantAgent, UserProxyAgent

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)
app.secret_key = os.urandom(24)

llm_config = {
    "config_list": [
        {
            "model": "llama-3.3-70b-versatile",
            "api_key": GROQ_API_KEY,
            "api_type": "openai",
            "base_url": "https://api.groq.com/openai/v1"
        }
    ],
    "temperature": 0
}

index_ready = False
index_lock = threading.Lock()


def is_termination_msg(msg):
    return msg.get("content") and "TERMINATE" in msg["content"]


def run_agent(query):
    legal_assistant = AssistantAgent(
        name="LegalAssistant",
        system_message=(
            "You are a helpful legal assistant that answers user queries ONLY by calling the 'retrieve_legal_context' tool. "
            "After answering, always respond with 'TERMINATE' to end the chat."
        ),
        llm_config=llm_config,
    )

    user = UserProxyAgent(
        name="User",
        llm_config=False,
        human_input_mode="NEVER",
        is_termination_msg=is_termination_msg,
        code_execution_config={"use_docker": False}
    )

    legal_assistant.register_for_llm(
        name="retrieve_legal_context",
        description="Retrieve relevant legal context from the indexed legal documents based on the query."
    )(retrieve_legal_context)

    user.register_for_execution(
        name="retrieve_legal_context"
    )(retrieve_legal_context)

    chat_result = user.initiate_chat(
        legal_assistant,
        message=query,
        summary_method="reflection_with_llm"
    )

    # Primary: use the LLM-generated summary
    summary = getattr(chat_result, "summary", None)
    if summary and summary.strip():
        return summary.strip()

    # Fallback: scan history for any assistant content with substance
    history = getattr(chat_result, "chat_history", []) or []
    for msg in reversed(history):
        content = msg.get("content") or ""
        role = msg.get("role", "")
        content_clean = content.replace("TERMINATE", "").strip()
        # Skip empty, tool-call jsons, or pure TERMINATE messages
        if content_clean and role == "assistant" and not content_clean.startswith("{"):
            return content_clean

    return "The assistant did not return a usable answer. Please try rephrasing your question."


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global index_ready
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "" or not file.filename.endswith(".pdf"):
        return jsonify({"error": "Please upload a valid PDF file"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        build_index_from_pdf(tmp_path, persist_dir="rag_faiss_store")
        with index_lock:
            index_ready = True
        return jsonify({"message": "PDF indexed successfully. You can now ask questions."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/chat", methods=["POST"])
def chat():
    global index_ready
    with index_lock:
        ready = index_ready

    if not ready:
        return jsonify({"error": "No document indexed yet. Please upload a PDF first."}), 400

    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    try:
        answer = run_agent(query)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
