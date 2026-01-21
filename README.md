# 🧪 IST 488 Labs: Document QA & Summarization

A multi-page Streamlit application for IST 488 labs.

## Features
- **Lab 1: Document QA**: Upload a document and ask questions about it using OpenAI's GPT models.
- **Lab 2: Document Summarization**: Upload a text or PDF file and generate summaries with customizable length and language.

## Setup

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key**
   Create a `.streamlit/secrets.toml` file:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```

3. **Run the App**
   ```bash
   streamlit run streamlit_app.py
   ```
