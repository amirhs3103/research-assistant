import streamlit as st
from model import LLM_Chat
from dotenv import load_dotenv
from uuid import uuid4
import pdfplumber

load_dotenv()

def extract_pdf_text(uploaded_pdf):
    if uploaded_pdf is not None:
        with pdfplumber.open(uploaded_pdf) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    return ""

class App:
    def __init__(self):
        st.set_page_config(page_title="AI Research Assistant", page_icon="📚")
        st.title("📘 AI Research Assistant")

        self.setup_sessions()

    def setup_sessions(self):
        if 'chat_sessions' not in st.session_state:
            st.session_state.chat_sessions = {"default": []}

        if 'current_session' not in st.session_state:
            st.session_state.current_session = "default"

        with st.sidebar:
            st.subheader("💬 Chat Sessions")

            session_ids = list(st.session_state.chat_sessions.keys())
            if session_ids:
                selected = st.selectbox(
                    "Select a session",
                    session_ids,
                    index=session_ids.index(st.session_state.current_session)
                    if st.session_state.current_session in session_ids else 0
                )
                if selected != st.session_state.current_session:
                    st.session_state.current_session = selected

            if st.button("➕ Create New Chat"):
                new_id = f"chat_{len(st.session_state.chat_sessions) + 1}_{str(uuid4())[:4]}"
                st.session_state.chat_sessions[new_id] = []
                st.session_state.current_session = new_id

            if st.session_state.current_session != "default":
                if st.button("🗑️ Delete This Chat"):
                    del st.session_state.chat_sessions[st.session_state.current_session]
                    st.session_state.current_session = "default"

            st.divider()

            st.subheader("📄 PDF Tools")
            uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
            pdf_text = extract_pdf_text(uploaded_pdf) if uploaded_pdf else None

            if pdf_text:
                st.success("PDF uploaded and processed.")

                if st.button("📃 Summarize PDF"):
                    st.session_state.pdf_action = "summarize"
                    st.session_state.pdf_text = pdf_text

                user_question = st.text_input("❓ Ask a question about the PDF")
                if user_question:
                    st.session_state.pdf_action = "question"
                    st.session_state.pdf_question = user_question
                    st.session_state.pdf_text = pdf_text

    def display_chat(self, chat_history):
        if len(chat_history) == 0 or st.sidebar.button("🔄 Reset Chat"):
            return True

        for msg in chat_history:
            if msg.type == 'AIMessageChunk':
                msg.type = 'ai'
            st.chat_message(msg.type).write(msg.content)
        return False

    def get_user_input(self):
        return st.chat_input("Ask something...")

    def display_message(self, message_type, content):
        if message_type == "human":
            st.chat_message(message_type).write(content)
        else:
            with st.chat_message("ai"):
                st.write_stream(content)

    def display_app(self):
        session_id = st.session_state.current_session
        backend = LLM_Chat(session_key=session_id)

        if self.display_chat(backend.get_chat_history()):
            backend.reset_chat()

        if "pdf_action" in st.session_state and "pdf_text" in st.session_state:
            if st.session_state.pdf_action == "summarize":
                self.display_message("human", "[Summarize the uploaded PDF]")
                response = backend.process_input(f"Summarize this document:\n\n{st.session_state.pdf_text[:4000]}")
                self.display_message("ai", response)
            elif st.session_state.pdf_action == "question":
                question = st.session_state.pdf_question
                self.display_message("human", question)
                response = backend.process_input(f"Based on this document:\n\n{st.session_state.pdf_text[:4000]}\n\nAnswer this question:\n{question}")
                self.display_message("ai", response)

            del st.session_state.pdf_action
            del st.session_state.pdf_text
            if "pdf_question" in st.session_state:
                del st.session_state.pdf_question

        prompt = self.get_user_input()
        if prompt:
            self.display_message("human", prompt)
            response = backend.process_input(prompt)
            self.display_message("ai", response)

if __name__ == "__main__":
    app = App()
    app.display_app()
