import streamlit as st
from openai import OpenAI
import PyPDF2
import os
import chromadb

# Title
st.title("Lab 4: RAG Pipeline with Vector DB")

# API Key access
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("OpenAI API key not found. Please configure it in .streamlit/secrets.toml", icon="🗝️")
    st.stop()

client = OpenAI(api_key=openai_api_key)

def create_vector_db():
    """Created and returns a ChromaDB collection with the PDF data."""
    # Define the directory containing the PDFs
    pdf_dir = "Lab-04-Data"
    
    # Check if directory exists
    if not os.path.exists(pdf_dir):
        st.error(f"Directory {pdf_dir} not found.")
        return None

    # Retrieve all PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        st.error(f"No PDF files found in {pdf_dir}.")
        return None
        
    # Initialize ChromaDB client
    chroma_client = chromadb.Client()
    
    # Create or get collection
    try:
        collection = chroma_client.create_collection(name="Lab4Collection")
    except Exception:
         # In case it already exists in this ephemeral session (unlikely with this logic but good practice)
        collection = chroma_client.get_collection(name="Lab4Collection")

    documents = []
    ids = []
    metadatas = []

    # Process each PDF
    for filename in pdf_files:
        file_path = os.path.join(pdf_dir, filename)
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                
                documents.append(text)
                ids.append(filename)
                metadatas.append({"filename": filename})
        except Exception as e:
            st.error(f"Error reading {filename}: {e}")

    # Add documents to collection
    # ChromaDB handles embeddings automatically if no embedding function is provided? 
    # The prompt asked to "For the embeddings vector use an OpenAI embeddings model". 
    # ChromaDB's default is all-MiniLM-L6-v2. We need to specify OpenAI.
    # However, for simplicity in this specific lab context without creating a custom EmbeddingFunction class overhead 
    # unless required by the library version, we might need to do manual embedding or use Chroma's utility.
    # Let's use OpenAI embeddings manually to ensure we meet the "For the embeddings vector use an OpenAI embeddings model" requirement explicitly.
    
    # Actually, let's use the embedding function feature if possible, or just pre-calculate.
    # Given the instructions "For the embeddings vector use an OpenAI embeddings model", let's do it manually 
    # to be safe and clear, or use the openai client we already have.
    
    # Generate embeddings
    embeddings = []
    for doc in documents:
        response = client.embeddings.create(input=doc, model="text-embedding-3-small")
        embeddings.append(response.data[0].embedding)

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    return collection

# Initialize Vector DB
if "Lab4_VectorDB" not in st.session_state:
    with st.spinner("Creating Vector Database from PDFs..."):
        st.session_state.Lab4_VectorDB = create_vector_db()

# --- Part A: Test the vectorDB ---
# Use a search string as a test and output an ordered list of 3 returned documents
st.subheader("Part A: Test Vector DB")
test_query = "Generative AI"
st.write(f"Testing with query: **{test_query}**")

if st.session_state.Lab4_VectorDB:
    # Generate embedding for the query
    query_response = client.embeddings.create(input=test_query, model="text-embedding-3-small")
    query_embedding = query_response.data[0].embedding
    
    # Query the collection
    results = st.session_state.Lab4_VectorDB.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    st.write("Top 3 Documents:")
    if results['ids']:
        for i, doc_id in enumerate(results['ids'][0]):
            st.write(f"{i+1}. {doc_id}")


# --- Chatbot Interface (Part B placeholder) ---
# Initialize Session State
if "lab4_messages" not in st.session_state:
    st.session_state.lab4_messages = []

# Display Chat History
for message in st.session_state.lab4_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about the course materials"):
    # Add user message to history
    st.session_state.lab4_messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # TODO: Implement RAG for Part B here
    # Placeholder response
    response = "I am a placeholder. RAG implementation coming in Part B."
    
    with st.chat_message("assistant"):
        st.markdown(response)
    
    st.session_state.lab4_messages.append({"role": "assistant", "content": response})
