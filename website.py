import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

# ---------- ENV & MODEL SETUP ----------
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="moonshotai/Kimi-K2-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# ---------- LLM QUERY FUNCTION ----------
def query_llm(history):
    system_prompt = (
        "You are a witty and culturally aware assistant trying to guess which Indian state a person is from.\n\n"
        "You must ask fun, creative, indirect questions based on:\n"
        "- Regional foods (e.g., dosa, momos, litti chokha)\n"
        "- Cultural habits (e.g., waking up early, festive dressing, hand gestures)\n"
        "- Local jokes and stereotypes (light-hearted only)\n"
        "- Popular celebrities, movies, festivals, or weather\n"
        "- Language quirks or expressions (without naming the language)\n"
        "- Common biases or preferences (like tea vs coffee, rice vs wheat)\n\n"
        "Every question MUST be answerable with only one of these:\n"
        "- Yes\n"
        "- No\n"
        "- Maybe\n"
        "- Don't Know\n\n"
        "**Never ask direct questions** like 'Are you from X?' or 'Do you live in Y?'. Be clever and subtle.\n\n"
        "At each step, do exactly one of the following:\n"
        "1. If you're confident, reply only as: GUESS: <state>\n"
        "2. Otherwise, reply only as: QUESTION: <your yes/no/maybe/don't know question>\n\n"
        "NEVER include anything else besides one of those two formats."
    )

    messages = [SystemMessage(content=system_prompt)]

    for q, a in history:
        messages.append(HumanMessage(content=q))
        messages.append(AIMessage(content=a))

    messages.append(HumanMessage(content="What's your next step?"))

    response = model.invoke(messages)
    content = response.content.strip()

    if content.startswith("GUESS:"):
        return None, content.replace("GUESS:", "").strip()
    elif content.startswith("QUESTION:"):
        return content.replace("QUESTION:", "").strip(), None
    else:
        return None, None

# ---------- STREAMLIT CONFIG & CSS ----------
st.set_page_config(page_title="ICG - Guess your state", layout="centered")

st.markdown("""
    <style>
    html, body, [class*="css"] {
        background: linear-gradient(to right, #f3f9ff, #e4f7ec);
        font-family: 'Segoe UI', sans-serif;
        text-align: center;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        font-size: 1.3em;
        padding: 0.75em 2em;
        margin: 0.4em;
        border: none;
        border-radius: 12px;
        width: 90%;
        background-image: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: bold;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-image: linear-gradient(135deg, #5a67d8, #6b46c1);
        transform: scale(1.03);
    }
    .restart-btn > button {
        background-image: linear-gradient(135deg, #ff416c, #ff4b2b) !important;
        color: white !important;
        font-weight: bold;
    }
    .restart-btn > button:hover {
        background-image: linear-gradient(135deg, #ff4b2b, #ff416c) !important;
        transform: scale(1.03);
    }
    </style>
""", unsafe_allow_html=True)

# ---------- INITIAL STATE ----------
if "page" not in st.session_state:
    st.session_state.page = "start"
    st.session_state.history = []
    st.session_state.guess = None

# ---------- RESTART FUNCTION ----------
def restart():
    st.session_state.page = "start"
    st.session_state.history = []
    st.session_state.guess = None
    st.rerun()

# ---------- LLM WRAPPER ----------
def get_llm_response(history):
    return query_llm(history)

# ---------- START PAGE ----------
def start_page():
    st.title("🌍 ICG - Guess Your State!")
    st.markdown("We'll try to guess which **Indian state** you're from based on your answers.")
    st.markdown("#### Ready to play?")

    if st.button("Start the Game"):
        st.session_state.page = "question"
        st.session_state.history = []
        st.session_state.guess = None

        with st.spinner("🚦 Starting..."):
            question, _ = get_llm_response([])

        if question:
            st.session_state.current_question = question
            st.rerun()
        else:
            st.error("Failed to load the first question. Please try again.")
            restart()

# ---------- QUESTION PAGE ----------
def question_page():
    st.markdown(f"## 🤔 {st.session_state.current_question}")
    st.markdown("### Choose your answer:")

    cols = st.columns(2)
    options = ["Yes", "No", "Maybe", "Don't Know"]

    def handle_answer(answer):
        st.session_state.history.append((st.session_state.current_question, answer))

        with st.spinner("🤖 Thinking..."):
            next_question, guessed_state = get_llm_response(st.session_state.history)

        if next_question is None and guessed_state is None:
            st.error("❌ Couldn't understand model response. Please restart.")
            return

        if guessed_state:
            st.session_state.page = "result"
            st.session_state.guess = guessed_state
        else:
            st.session_state.current_question = next_question
            st.session_state.page = "question"

        st.rerun()

    for i, option in enumerate(options):
        if cols[i % 2].button(option):
            handle_answer(option)

    st.markdown("---")
    with st.container():
        with st.container():
            st.markdown('<div class="restart-btn">', unsafe_allow_html=True)
            if st.button("Restart"):
                restart()
            st.markdown('</div>', unsafe_allow_html=True)

# ---------- RESULT PAGE ----------
def result_page():
    st.markdown("## 🎉 Our guess is...")
    st.success(f"🏁 YOU ARE FROM **{st.session_state.guess}**!")
    st.markdown("Thanks for playing this fun game with us! 💫")

    st.markdown("---")
    st.markdown('<div class="restart-btn">', unsafe_allow_html=True)
    if st.button("Play Again"):
        restart()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- PAGE ROUTING ----------
if st.session_state.page == "start":
    start_page()
elif st.session_state.page == "question":
    question_page()
elif st.session_state.page == "result":
    result_page()