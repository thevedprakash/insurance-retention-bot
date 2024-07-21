import streamlit as st
import pandas as pd
from bot.gpt_agent import GPT
from bot.utils import load_customer_data
from langchain_google_genai import ChatGoogleGenerativeAI
import random
import time

def initialize_agent():
    llm = ChatGoogleGenerativeAI(model='gemini-1.0-pro', temperature=0.2)
    gpt_agent = GPT.from_llm(llm=llm, verbose=True)
    return gpt_agent


def main():
    gpt_agent = initialize_agent()

    print("Starting conversation with GPT Agent. Type 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        gpt_agent.human_step(user_input)
        bot_response = gpt_agent.step()
        print(f"Sophia: {bot_response}")

if __name__ == "__main__":
    main()