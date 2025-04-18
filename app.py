import streamlit as st
from agent import run_agent

st.set_page_config(page_title="LLM Agent")

st.title("🤖 LLM-Powered Agent")

user_input = st.text_input("Ask something:")

if user_input:
    with st.spinner("Thinking..."):
        response = run_agent(user_input)
        st.write(response)
