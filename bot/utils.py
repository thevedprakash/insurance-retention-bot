import pandas as pd
from gpt_agent import GPT
from langchain.llms import ChatGoogleGenerativeAI

def load_customer_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def initialize_agent():
    llm = ChatGoogleGenerativeAI(model='gemini-1.0-pro', temperature=0.2)
    gpt_agent = GPT.from_llm(llm=llm, verbose=True)
    return gpt_agent
