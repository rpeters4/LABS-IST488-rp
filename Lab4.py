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

# Helper function for chunking text
def chunk_text(text, chunk_size=1000, overlap=200):
    """Splits text into chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

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
    collection_name = "Lab4Collection" 
    
    try:
        # Try to delete if it exists to ensure freshness (optional, but good for dev)
        # Note: In production you wouldn't delete it every time, but for this lab 
        # where we might be actively changing the logic, it helps.
        # However, the requirement says "Only create the ChromaDB once".
        # So we should rely on the st.session_state check.
        # But since we are inside create_vector_db which is only called if not in session state...
        # Let's just use get_or_create to be safe.
        collection = chroma_client.get_or_create_collection(name=collection_name)
    except:
        collection = chroma_client.create_collection(name=collection_name)

    documents = []
    ids = []
    metadatas = []
    
    id_counter = 0

    # Process each PDF
    for filename in pdf_files:
        file_path = os.path.join(pdf_dir, filename)
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                full_text = ""
                for page in reader.pages:
                    full_text += page.extract_text()
                
                # CHUNK the text
                chunks = chunk_text(full_text, chunk_size=1000, overlap=200)
                
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    ids.append(f"{filename}_chunk_{i}") # Unique ID per chunk
                    metadatas.append({"filename": filename, "chunk_id": i})
                    
        except Exception as e:
            st.error(f"Error reading {filename}: {e}")

    # Generate embeddings
    # Using batches to avoid hitting API limits if many chunks
    embeddings = []
    batch_size = 100 # OpenAI limit is often higher, but 100 is safe
    
    progress_bar = st.progress(0, text="Generating embeddings...")
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        try:
            response = client.embeddings.create(input=batch_docs, model="text-embedding-3-small")
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
            progress_bar.progress((i + len(batch_docs)) / len(documents), text=f"Generated {i + len(batch_docs)}/{len(documents)} embeddings")
        except Exception as e:
            st.error(f"Error generating embeddings for batch {i}: {e}")
            return None # Stop if embedding fails
            
    progress_bar.empty()

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

# --- Chatbot Interface (Part B) ---
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

    # Retrieve relevant documents
    context_text = ""
    retrieved_docs = []
    
    if "Lab4_VectorDB" in st.session_state and st.session_state.Lab4_VectorDB:
        with st.spinner("Retrieving relevant information..."):
            query_response = client.embeddings.create(input=prompt, model="text-embedding-3-small")
            query_embedding = query_response.data[0].embedding
            
            # Increased n_results since we are using chunks now
            results = st.session_state.Lab4_VectorDB.query(
                query_embeddings=[query_embedding],
                n_results=5  # Retrieve top 5 chunks
            )
            
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    filename = results['metadatas'][0][i]['filename']
                    context_text += f"\n\n--- Document Snippet {i+1} (Source: {filename}) ---\n{doc}"
                    retrieved_docs.append(filename)

    # Prepare messages for LLM
    system_prompt = (
        "You are a helpful teaching assistant for a university course. "
        "You have access to the following course materials (syllabi, etc.) to answer student questions. "
        "Use the provided context to answer the user's question. "
        "If the answer is found in the context, please cite the document or mention that you found it in the course materials. "
        "If the answer is NOT in the context, utilize your general knowledge but clearly state that this information is not from the specific course documents provided."
        f"\n\nContext Information:{context_text}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    
    messages.append({"role": "user", "content": prompt})


    # Generate Response
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        )
        response = st.write_stream(stream)
    
    st.session_state.lab4_messages.append({"role": "assistant", "content": response})
    
    # DEBUG SIDEBAR (Shows info for the LAST query)
    st.session_state.last_context_length = len(context_text)
    st.session_state.last_retrieved_docs = retrieved_docs

# Debug Sidebar Logic
with st.sidebar:
    st.header("Debug Tool")
    if "last_context_length" in st.session_state:
        st.write(f"**Context Length:** {st.session_state.last_context_length} chars")
    
    if "last_retrieved_docs" in st.session_state:
        st.write("**Retrieved Documents:**")
        # De-duplicate filenames for display while keeping order
        seen = set()
        seen_add = seen.add
        unique_docs = [x for x in st.session_state.last_retrieved_docs if not (x in seen or seen_add(x))]
        
        for doc in unique_docs:
            st.write(f"- {doc}")
            
    if st.button("Clear Conversation"):
        st.session_state.lab4_messages = []
        st.rerun()
