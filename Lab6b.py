import os
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ──────────────────────────────────────────────
# API Key Setup
# ──────────────────────────────────────────────
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("OpenAI API key not found. Please add OPENAI_API_KEY to .streamlit/secrets.toml")
    st.stop()

# ──────────────────────────────────────────────
# Initialize LLM — OpenAI GPT-4o via LangChain
# ──────────────────────────────────────────────
llm = init_chat_model("gpt-4o", model_provider="openai")

# ──────────────────────────────────────────────
# Page Setup
# ──────────────────────────────────────────────
st.title("Lab 6b: Movie Recommendation Chatbot 🎬")
st.caption("🎥 Powered by LangChain & OpenAI — get personalized movie picks and ask follow-up questions!")

# ──────────────────────────────────────────────
# Sidebar — Genre, Mood, Persona selectors
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("🎬 Movie Preferences")

    genre = st.selectbox(
        "Genre",
        ["Action", "Comedy", "Horror", "Drama", "Sci-Fi", "Thriller", "Romance"],
    )

    mood = st.selectbox(
        "Mood",
        ["Excited", "Happy", "Sad", "Bored", "Scared", "Romantic", "Curious", "Tense", "Melancholy"],
    )

    persona = st.selectbox(
        "Persona",
        ["Film Critic", "Casual Friend", "Movie Journalist"],
    )

# ──────────────────────────────────────────────
# Part B: Recommendation Chain
# ──────────────────────────────────────────────

# Prompt template for movie recommendations
recommendation_template = PromptTemplate(
    input_variables=["genre", "mood", "persona"],
    template=(
        "You are a {persona}. Recommend exactly 3 movies in the {genre} genre "
        "that would be perfect for someone who is feeling {mood}. "
        "For each movie, include the title, release year, and a short reason why "
        "it fits. Match the tone and style of a {persona} in your response."
    ),
)

# Build the recommendation chain: prompt → model → parser
recommendation_chain = recommendation_template | llm | StrOutputParser()

# Session state for storing the last recommendation
if "last_recommendation" not in st.session_state:
    st.session_state.last_recommendation = None

# Generate recommendations on button click
if st.button("🎬 Get Recommendations", type="primary"):
    with st.spinner("Finding the perfect movies for you…"):
        result = recommendation_chain.invoke({
            "genre": genre,
            "mood": mood,
            "persona": persona,
        })
    st.session_state.last_recommendation = result

# Display the stored recommendation
if st.session_state.last_recommendation:
    st.markdown(st.session_state.last_recommendation)

# ──────────────────────────────────────────────
# Part C: Follow-Up Chain
# ──────────────────────────────────────────────
st.divider()
follow_up = st.text_input("Ask a follow-up question about these movies:")

# Prompt template for follow-up questions
followup_template = PromptTemplate(
    input_variables=["recommendations", "question"],
    template=(
        "Here are some movie recommendations that were previously given:\n\n"
        "{recommendations}\n\n"
        "The user has a follow-up question: {question}\n\n"
        "Please answer the question based on the recommendations above. "
        "Be helpful and specific."
    ),
)

# Build the follow-up chain: prompt → model → parser
followup_chain = followup_template | llm | StrOutputParser()

if follow_up:
    if st.session_state.last_recommendation is None:
        st.warning("Please get recommendations first before asking a follow-up question.")
    else:
        with st.spinner("Looking into that…"):
            followup_result = followup_chain.invoke({
                "recommendations": st.session_state.last_recommendation,
                "question": follow_up,
            })
        st.markdown(followup_result)
