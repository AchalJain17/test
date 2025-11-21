# streamlit_app.py
import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv("config.env")
BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Al-Powered QC Assistant - Upload Excel")
st.title("AI QC Assistant — Upload dataset (Excel)")

upload_folder = os.getenv("UPLOAD_FOLDER", "./uploaded_excels")
os.makedirs(upload_folder, exist_ok=True)

uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx", "xls"])
text_columns_input = st.text_input("Text columns in the Excel file (comma separated)", value="defect_description,process_parameters,inspection_result")

if uploaded_file:
    file_path = os.path.join(upload_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved file to {file_path}")

    if st.button("Send to backend & Index"):
        with st.spinner("Uploading and indexing..."):
            files = {"file": open(file_path, "rb")}
            data = {"text_columns": text_columns_input}
            resp = requests.post(f"{BACKEND_URL}/ingest", files=files, data=data)
            st.write(resp.json())

st.markdown("---")
st.header("Ask the indexed data")
question = st.text_input("Ask a question about defects / batches / root causes")
use_browser = st.checkbox("Use browser agent (LangGraph) fallback for web search (if enabled in backend)", value=False)

if st.button("Ask"):
    if not question:
        st.warning("Write a question first.")
    else:
        payload = {"question": question, "use_browser": use_browser}
        resp = requests.post(f"{BACKEND_URL}/query", json=payload)
        try:
            data = resp.json()
            if data.get("status") == "success":
                st.subheader("Answer")
                st.write(data.get("answer"))
                st.subheader("Sources (snippets)")
                for s in data.get("sources", []):
                    st.write(s.get("page_content"))
            else:
                st.error(data)
        except Exception as e:
            st.error(f"Error calling backend: {e}")
