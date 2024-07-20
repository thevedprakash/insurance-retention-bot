import streamlit as st
import pandas as pd
from gpt_agent import GPT
from utils import load_customer_data, initialize_agent

def main():
    st.title("Insurance Customer Retention Bot")
    st.write("Upload your customer data and start the conversation")

    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file is not None:
        data = load_customer_data(uploaded_file)
        st.write(data.head())

        if st.button("Start Conversation"):
            handle_conversation(data)

def handle_conversation(data):
    customer = data.iloc[0]

    conversation_history = []
    conversation_stage = '1'
    gpt_agent = initialize_agent()

    st.write(f"Starting conversation with {customer['First Name']} {customer['Last Name']}...")

    while conversation_stage != '7':
        prompt = gpt_agent.conversation_stage_dict[conversation_stage]
        st.write(prompt)
        response = st.text_input("Customer Response", key=f"stage_{conversation_stage}")

        if st.button("Next Stage"):
            conversation_history.append((prompt, response))
            gpt_agent.human_step(response)
            conversation_stage = gpt_agent.determine_conversation_stage()

            ai_message = gpt_agent.step()
            st.write(f"Agent: {ai_message}")

    st.write("End of Conversation")

    st.write("Conversation History:")
    for stage, (prompt, response) in enumerate(conversation_history):
        st.write(f"Stage {stage+1}:")
        st.write(f"Bot: {prompt}")
        st.write(f"Customer: {response}")

if __name__ == "__main__":
    main()
