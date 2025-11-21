# backend/app.py
import os
from flask import Flask, request, jsonify
from rag_indexer import build_rag_index, query_rag
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "config.env"))
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploaded_excels")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_index")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

@app.route("/ingest", methods=["POST"])
def ingest():
    """
    Accepts multipart form with file + text_columns (comma separated) or JSON body with path.
    """
    try:
        # support direct file upload
        if "file" in request.files:
            f = request.files["file"]
            if f.filename == "":
                return jsonify({"status": "error", "message": "No file selected"}), 400
            save_path = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(save_path)
        else:
            # optionally accept path in JSON
            data = request.get_json() or {}
            save_path = data.get("path")
            if not save_path or not os.path.exists(save_path):
                return jsonify({"status": "error", "message": "No file and no valid path provided"}), 400

        text_columns = request.form.get("text_columns") or (request.json and request.json.get("text_columns"))
        if not text_columns:
            # default heuristic columns used by this defect dataset
            text_columns = "defect_description,process_parameters,inspection_result"
        text_columns = [c.strip() for c in text_columns.split(",")]

        # Build the RAG index
        build_rag_index(save_path, text_columns, persist_dir=CHROMA_PERSIST_DIR)
        return jsonify({"status": "success", "message": "File ingested and indexed."}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json() or {}
    question = data.get("question")
    if not question:
        return jsonify({"status": "error", "message": "Provide 'question' in JSON body"}), 400
    try:
        result = query_rag(question, persist_dir=CHROMA_PERSIST_DIR)
        return jsonify({"status": "success", "answer": result["answer"], "sources": result["sources"]}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8000)
