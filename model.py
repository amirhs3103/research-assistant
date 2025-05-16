from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, ConversationalAgent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()

wiki = WikipediaAPIWrapper()

tools = [
    SemanticScholarQueryRun(),
    WikipediaQueryRun(api_wrapper=wiki),
    ArxivQueryRun(),
    DuckDuckGoSearchRun()
]

class AgentRunnable(Runnable):
    def __init__(self, agent_executor):
        self.agent_executor = agent_executor

    def invoke(self, input, config=None):
        return self.agent_executor.run(input)

    def stream(self, input, config=None):
        return self.agent_executor.stream(input)

class LLM_Chat:
    def __init__(self, session_key="default"):
        self.session_key = session_key

        self.chat_history = StreamlitChatMessageHistory(key=session_key)
        self.memory = ConversationBufferMemory(
            chat_memory=self.chat_history,
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )

        self.llm = ChatOpenAI(model='gpt-4o-mini', streaming=True)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful expert research assistant. Use your tools to answer questions and provide sources. When relevant, search the internet for recent information. use semantic scholar and arxiv for research papers, and Wikipedia for general knowledge."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        agent = ConversationalAgent.from_llm_and_tools(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            output_key="output"
        )

        self.chain_with_message_history = RunnableWithMessageHistory(
            AgentRunnable(self.agent_executor),
            lambda session_id: self.chat_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

    def reset_chat(self):
        self.chat_history.clear()

    def get_chat_history(self):
        return self.chat_history.messages

    def process_input(self, prompt):
        config = {"configurable": {"session_id": self.session_key}}
        try:
            inputs = {
                "input": prompt,
                "chat_history": self.memory.load_memory_variables({})["chat_history"]
            }

            def stream_output_only():
                for chunk in self.chain_with_message_history.stream(inputs, config):
                    if isinstance(chunk, dict) and "output" in chunk:
                        yield chunk["output"]

            return stream_output_only()
        except Exception as error:
            return str(error)

