import streamlit as st
from openai import OpenAI
from pydantic import BaseModel

# ──────────────────────────────────────────────
# Structured output model (Part D)
# ──────────────────────────────────────────────
class ResearchSummary(BaseModel):
    main_answer: str
    key_facts: list[str]
    source_hint: str

# ──────────────────────────────────────────────
# Page Setup
# ──────────────────────────────────────────────
st.title("Lab 6: OpenAI Responses API Agent 🔍")
st.caption("🌐 This agent has **web search** enabled — it can find up-to-date information and cite sources.")

# ──────────────────────────────────────────────
# Sidebar Controls
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("Agent Settings")
    use_structured = st.checkbox("Return structured summary", help="Parse the response into main_answer, key_facts, and source_hint.")
    use_streaming = st.checkbox("Enable streaming", help="Stream tokens to the UI in real-time.")

# ──────────────────────────────────────────────
# OpenAI Client
# ──────────────────────────────────────────────
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("OpenAI API key not found. Please add OPENAI_API_KEY to .streamlit/secrets.toml")
    st.stop()

client = OpenAI(api_key=api_key)

MODEL = "gpt-4o"
INSTRUCTIONS = "You are a helpful research assistant. Cite your sources when possible."
TOOLS = [{"type": "web_search_preview"}]

# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────
if "last_response_id" not in st.session_state:
    st.session_state.last_response_id = None
if "first_response_text" not in st.session_state:
    st.session_state.first_response_text = None
if "followup_response_text" not in st.session_state:
    st.session_state.followup_response_text = None

# ──────────────────────────────────────────────
# Helper: make a Responses API call
# ──────────────────────────────────────────────
def _call_responses_api(user_input: str, previous_response_id: str | None = None):
    """
    Makes a call via the Responses API.
    Handles structured output, streaming, and plain modes.
    Returns (display_text, response_id).
    """
    common_kwargs = dict(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input=user_input,
        tools=TOOLS,
        previous_response_id=previous_response_id,
    )

    # ── Structured output mode ──
    if use_structured:
        response = client.responses.parse(
            **common_kwargs,
            text_format=ResearchSummary,
        )
        return response, response.id

    # ── Streaming mode ──
    if use_streaming:
        return "STREAM", common_kwargs  # handled by caller

    # ── Normal mode ──
    response = client.responses.create(**common_kwargs)
    return response.output_text, response.id


def _display_structured(response):
    """Render a structured ResearchSummary response."""
    # The parsed output lives on the last output item of type "message"
    for item in response.output:
        if item.type == "message":
            for content_block in item.content:
                if hasattr(content_block, "parsed") and content_block.parsed is not None:
                    summary = content_block.parsed
                    st.subheader("Main Answer")
                    st.write(summary.main_answer)
                    st.subheader("Key Facts")
                    for fact in summary.key_facts:
                        st.markdown(f"- {fact}")
                    st.caption(f"📌 Source hint: {summary.source_hint}")
                    return
    # Fallback: show raw text
    st.write(response.output_text)


# ──────────────────────────────────────────────
# Part A: Initial Question
# ──────────────────────────────────────────────
st.subheader("Ask a question")
user_question = st.text_input("Enter your question:", key="question_input", placeholder="e.g. What are the latest developments in quantum computing?")

if st.button("Submit", key="submit_question", type="primary"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching & thinking…"):
            result, meta = _call_responses_api(user_question)

        if result == "STREAM":
            # meta holds the kwargs dict
            stream = client.responses.create(**meta, stream=True)
            collected_text = ""
            response_id = None
            placeholder = st.empty()
            for event in stream:
                if event.type == "response.output_text.delta":
                    collected_text += event.delta
                    placeholder.markdown(collected_text)
                elif event.type == "response.completed":
                    response_id = event.response.id
            st.session_state.last_response_id = response_id
            st.session_state.first_response_text = collected_text
        elif use_structured:
            # result is the full response object
            _display_structured(result)
            st.session_state.last_response_id = meta  # meta is response.id
            st.session_state.first_response_text = "(structured)"
        else:
            st.markdown(result)
            st.session_state.last_response_id = meta  # meta is response.id
            st.session_state.first_response_text = result

        # Reset follow-up when a new first question is asked
        st.session_state.followup_response_text = None

# Show previous first response if it exists (survives reruns)
if st.session_state.first_response_text and st.session_state.first_response_text != "(structured)":
    if not user_question:  # only re-display on reruns where the button wasn't just pressed
        st.markdown("**Previous response:**")
        st.markdown(st.session_state.first_response_text)

# ──────────────────────────────────────────────
# Part B: Follow-Up Question
# ──────────────────────────────────────────────
if st.session_state.last_response_id:
    st.divider()
    st.subheader("Ask a follow-up question")
    followup = st.text_input(
        "Follow-up question:",
        key="followup_input",
        placeholder="e.g. Can you give me more detail on the second point?",
    )

    if st.button("Submit Follow-Up", key="submit_followup", type="primary"):
        if not followup.strip():
            st.warning("Please enter a follow-up question.")
        else:
            with st.spinner("Searching & thinking…"):
                result, meta = _call_responses_api(
                    followup,
                    previous_response_id=st.session_state.last_response_id,
                )

            if result == "STREAM":
                stream = client.responses.create(**meta, stream=True)
                collected_text = ""
                response_id = None
                placeholder = st.empty()
                for event in stream:
                    if event.type == "response.output_text.delta":
                        collected_text += event.delta
                        placeholder.markdown(collected_text)
                    elif event.type == "response.completed":
                        response_id = event.response.id
                st.session_state.last_response_id = response_id
                st.session_state.followup_response_text = collected_text
            elif use_structured:
                _display_structured(result)
                st.session_state.last_response_id = meta
                st.session_state.followup_response_text = "(structured)"
            else:
                st.markdown(result)
                st.session_state.last_response_id = meta
                st.session_state.followup_response_text = result

    # Show previous follow-up response on reruns
    if st.session_state.followup_response_text and st.session_state.followup_response_text != "(structured)":
        if not followup:
            st.markdown("**Previous follow-up response:**")
            st.markdown(st.session_state.followup_response_text)
