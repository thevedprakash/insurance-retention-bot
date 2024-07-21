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

import random
import time

def get_current_timestamp():
    return str(int(time.time() * 1000))

import random
import time

def get_current_timestamp():
    return str(int(time.time() * 1000))

def handle_conversation(data):
    customer = data.iloc[0]

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
        print(f"1. {st.session_state}")

    if 'conversation_stage' not in st.session_state:
        st.session_state.conversation_stage = '1'
        print(f"2. {st.session_state}")

    if 'gpt_agent' not in st.session_state:
        st.session_state.gpt_agent = initialize_agent()
        print(f"3. {st.session_state}")

    conversation_history = st.session_state.conversation_history
    conversation_stage = st.session_state.conversation_stage
    gpt_agent = st.session_state.gpt_agent

    print(f"Starting conversation with {customer['First Name']} {customer['Last Name']}...")

    if len(conversation_history) == 0:
        # Generate the initial prompt using the LLM
        ai_message = gpt_agent.step()
        st.write(f"Agent: {ai_message}")
        conversation_history.append(("Introduction", ai_message))
        st.session_state.conversation_history = conversation_history
        print(conversation_history)

    # Display the last agent message
    if conversation_history:
        print(f"Agent: {conversation_history[-1][1]}")
        # st.write(f"Agent: {conversation_history[-1][1]}")

    print("I'm here to print customer response.")
    # Use the current timestamp to generate a unique key
    response_key = f"stage_{conversation_stage}_{get_current_timestamp()}"

    # Create the text input widget for customer response
    # response = st.text_input("Customer Response", key=response_key)
    # response = "Tell me the benefit of the Policy. I don't get what is it for." # 'Value Proposition'
    # response = "I'm unhappy I submitted claim 5 times before but it get fulfilled just one. I'm frustrated with decline of claim request citing silly reasons." # 'Needs Analysis'
    response= "As I mentioned earlier, my claim has been processed only 1 out of 5 time. How this can dealt with is there any way to ensure my claim getting acceted." #'Needs Analysis:'
    # response = "I have seen another policy which deliver similar benefits why should I not use it." # 'Value Proposition'
    # response = "I would like to talk to customer care to discuss this further." # 'Introduction'
    # response = "I'm happy to continue making payment for policy premium." # 'Introduction'
    # response = "I'm Busy" # 'End Conversation' 
    # response = 'Hey Thanks for reaching, I made payment today. GoodBye.' # 'End Conversation'
    print(f"Customer Response: {response}")

    # Only display "Next Stage" button if there is user input
    if response:
        print(f"{response}")
        value = st.button("Next Stage", key=f"button_{get_current_timestamp()}")
        print(f"value: {value},")
        # if st.button("Next Stage", key=f"button_{get_current_timestamp()}"):
        print("1")
        st.write(f"Customer: {response}")
        print("2")
        conversation_history.append((conversation_stage, response))
        print("3")
        gpt_agent.human_step(response)
        conversation_stage = gpt_agent.determine_conversation_stage()
        print(f"New Conversation Stage: {conversation_stage}")

        ai_message = gpt_agent.step()
        st.write(f"Agent: {ai_message}")
        print("4")
        conversation_history.append((conversation_stage, ai_message))

        # Update session state
        st.session_state.conversation_history = conversation_history
        st.session_state.conversation_stage = conversation_stage
        print("5")

    st.write("Conversation History:")
    for stage, (prompt, response) in enumerate(conversation_history):
        st.write(f"Stage {stage+1}:")
        st.write(f"Conversation Category: {prompt}")
        st.write(f"Sophia: {response}")


# ---------------------------------------------------------

def main():
    st.title("Insurance Customer Retention Bot")
    st.write("Upload your customer data and start the conversation")

    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file is not None:
        data = load_customer_data(uploaded_file)
        st.write(data.head())

        if st.button("Start Conversation"):
            handle_conversation(data)

if __name__ == "__main__":
    main()
