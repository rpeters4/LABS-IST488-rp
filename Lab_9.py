import streamlit as st
from openai import OpenAI
import json
import os

# ── Page Title ──────────────────────────────────────────────────────────────
st.title("🧠 Chatbot with Long-Term Memory")

# ── API Client Setup ────────────────────────────────────────────────────────
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("OpenAI API key not found. Please configure it in .streamlit/secrets.toml", icon="🗝️")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# ── Memory File Path ────────────────────────────────────────────────────────
MEMORY_FILE = "memories.json"

# ── Part B: Memory System ──────────────────────────────────────────────────


def load_memories() -> list[str]:
    """Load memories from the JSON file. Returns an empty list if the file
    does not exist or is malformed."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def save_memories(memories: list[str]) -> None:
    """Write the list of memories to the JSON file."""
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=2)


# ── Sidebar: Display Memories ──────────────────────────────────────────────
st.sidebar.header("📝 Long-Term Memories")

memories = load_memories()

if memories:
    for i, memory in enumerate(memories, 1):
        st.sidebar.markdown(f"**{i}.** {memory}")
else:
    st.sidebar.info("No memories yet. Start chatting!")

if st.sidebar.button("🗑️ Clear All Memories"):
    save_memories([])
    st.rerun()

# ── Part C: Build the Chatbot ──────────────────────────────────────────────

# Initialize chat history in session state
if "lab9_messages" not in st.session_state:
    st.session_state.lab9_messages = []

# Display existing chat history
for message in st.session_state.lab9_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Say something…"):
    # Add user message to session state and display it
    st.session_state.lab9_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── 1) Inject memories into the system prompt ───────────────────────
    current_memories = load_memories()

    system_content = (
        "You are a friendly, helpful assistant. "
        "You have a long-term memory system that remembers facts about the user across conversations."
    )

    if current_memories:
        memory_block = "\n".join(f"- {m}" for m in current_memories)
        system_content += (
            "\n\nHere are things you remember about this user from past conversations:\n"
            + memory_block
            + "\n\nUse these memories naturally in your responses when relevant. "
            "For example, greet the user by name if you know it."
        )

    messages_to_send = [
        {"role": "system", "content": system_content},
        *st.session_state.lab9_messages,
    ]

    # ── Generate assistant response (streaming) ─────────────────────────
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_to_send,
            stream=True,
        )
        response = st.write_stream(stream)

    # Save assistant response to session state
    st.session_state.lab9_messages.append({"role": "assistant", "content": response})

    # ── 2) Extract new memories with a second LLM call ──────────────────
    existing_json = json.dumps(current_memories)

    extraction_prompt = (
        "You are a memory extraction assistant. Your ONLY job is to identify NEW facts "
        "about the user from the conversation below that are worth remembering long-term "
        "(e.g., name, location, major, university, hobbies, preferences, favorite foods, etc.).\n\n"
        f"Already-saved memories (do NOT duplicate these):\n{existing_json}\n\n"
        f"User message: {prompt}\n"
        f"Assistant response: {response}\n\n"
        "Return ONLY a JSON array of strings with any NEW facts. "
        "If there are no new facts, return an empty array: []\n"
        "Example output: [\"User's name is Alex\", \"User studies Data Science\"]\n"
        "Do NOT include any explanation — just the JSON array."
    )

    try:
        extraction_response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0,
        )

        raw = extraction_response.choices[0].message.content.strip()

        # Handle potential markdown code fences around JSON
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]  # remove first line (```json)
            raw = raw.rsplit("```", 1)[0]  # remove closing ```
            raw = raw.strip()

        new_facts: list[str] = json.loads(raw)

        if new_facts:
            updated = current_memories + new_facts
            save_memories(updated)
            st.rerun()  # refresh sidebar with new memories

    except (json.JSONDecodeError, Exception) as e:
        # Silently continue if extraction fails — the chat still works
        st.sidebar.warning(f"Memory extraction hiccup: {e}")
