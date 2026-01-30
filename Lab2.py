import streamlit as st
from openai import OpenAI
import os

# import pdf 
try:
    from PyPDF2 import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Show title and description.
st.title("Ryan's Lab 2: Document Summarizer")
st.write(
    "Upload a document below and get a summary using GPT. "
    "Select your preferred language, summary format, and model in the sidebar."
)

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Summary Options")
    
    # Language selection
    language = st.selectbox(
        "Output Language",
        options=["English", "Spanish", "French", "German", "Chinese", "Japanese"],
        index=0
    )
    
    # Summary type selection
    summary_type = st.selectbox(
        "Summary Format",
        options=[
            "100 words",
            "2 connecting paragraphs",
            "5 bullet points"
        ],
        index=0
    )
    
    # Model selection
    use_advanced_model = st.checkbox("Use advanced model (GPT-4o)", value=False)
    model = "gpt-4o" if use_advanced_model else "gpt-4o-mini"
    
    st.divider()
    st.caption(f"Using model: {model}")

# --- API Key from secrets ---
# Get API key from Streamlit secrets or environment variable
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API key not found. Please configure it in `.streamlit/secrets.toml` or as an environment variable.", icon="🔑")
    st.stop()

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# --- File Upload ---
if PDF_SUPPORT:
    uploaded_file = st.file_uploader(
        "Upload a document (.pdf, .txt, or .md)", type=("pdf", "txt", "md")
    )
else:
    st.warning("PyPDF2 not installed. PDF support disabled.")
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

def extract_text(file):
    """Extract text from uploaded file."""
    if file.name.endswith(".pdf") and PDF_SUPPORT:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        return file.read().decode()

if uploaded_file:
    # Extract text from the document
    document_text = extract_text(uploaded_file)
    
    if not document_text.strip():
        st.warning("Could not extract text from the document.")
        st.stop()
    
    st.success(f"Document loaded: {uploaded_file.name}")
    
    # Show a preview
    with st.expander("Preview Document Text"):
        st.text(document_text[:1000] + ("..." if len(document_text) > 1000 else ""))
    
    # Summarize button
    if st.button("Generate Summary", type="primary"):
        # Build the prompt based on user selections
        prompt = f"""Summarize the following document in {summary_type}.
Respond entirely in {language}.

---
{document_text}
"""
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        with st.spinner("Generating summary..."):
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                )
                
                st.subheader("Summary")
                st.write_stream(stream)
            except Exception as e:
                st.error(f"Error generating summary: {e}")
