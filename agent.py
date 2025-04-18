from langchain_openai.chat_models import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0.7, model_name='deepseek/deepseek-chat-v3-0324:free', streaming=True)


def run_agent(prompt):
    return llm.invoke(prompt)