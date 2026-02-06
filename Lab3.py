import streamlit as st
from openai import OpenAI

# Title
st.title("Lab 3: Chatbot with Conversational Memory")

# API Key access
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("OpenAI API key not found. Please configure it in .streamlit/secrets.toml", icon="🗝️")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Buffer Configuration
max_tokens = st.sidebar.slider("Max Tokens for Context", min_value=100, max_value=2000, value=500, step=100)

def get_num_tokens(text):
    """Returns an estimated number of tokens in a text string."""
    return int(len(text) / 4)

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("What up dawg?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # System Prompt implementation
    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful assistant designed to answer questions for a 10-year-old. "
            "Use simple language and clear explanations. "
            "After answering a question, ALWAYS ask: 'Do you want more info?' "
            "If the user says 'Yes', provide more details and ask 'Do you want more info?' again. "
            "If the user says 'No', respond with 'What else can I help you with?' and stop regarding the previous topic."
        )
    }

    # Buffer Logic: Token-based
    # Always include system prompt
    current_tokens = get_num_tokens(system_prompt["content"])
    messages_to_send = [system_prompt]
    
    # Let's simple traverse backwards and add until we hit the limit.
    temp_messages = []
    for message in reversed(st.session_state.messages):
        msg_tokens = get_num_tokens(message["content"])
        if current_tokens + msg_tokens <= max_tokens:
            temp_messages.insert(0, message) # Prepend
            current_tokens += msg_tokens
        else:
            break
            
    messages_to_send.extend(temp_messages)
    
    st.sidebar.write(f"Tokens in context: {current_tokens}")

    # Generate Stream Response
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_to_send,
            stream=True,
        )
        response = st.write_stream(stream)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
