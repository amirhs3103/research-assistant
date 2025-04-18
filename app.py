import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_openai.chat_models import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

load_dotenv()

llm = ChatOpenAI(temperature=0.7, model_name='deepseek/deepseek-chat-v3-0324:free', streaming=True)
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, [tool], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[tool], verbose=True)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke(
            {"input": prompt}, {"callbacks": [st_callback]}
        )
        st.write(response["output"])
