import pandas as pd
from bot.gpt_agent import GPT, StageAnalyzerChain, ConversationChain
from bot.rag_gpt_agent import RAGGPT
from bot.document_processor import initialize_document_store, retrieve_relevant_documents
from langchain_google_genai import ChatGoogleGenerativeAI

def initialize_agent():
    llm = ChatGoogleGenerativeAI(model='gemini-1.0-pro', temperature=0.2)
    stage_analyzer_chain = StageAnalyzerChain.from_llm(llm=llm, verbose=True)
    conversation_utterance_chain = ConversationChain.from_llm(llm=llm, verbose=True)
    
    # Initialize the document store and load the document
    pdf_path = 'data/Principal-Sample-Life-Insurance-Policy.pdf'
    document_store = initialize_document_store(pdf_path)
    
    rag_gpt_agent = RAGGPT(
        document_store=document_store, 
        retriever=lambda context: retrieve_relevant_documents(context, document_store=document_store, top_n=1), 
        stage_analyzer_chain=stage_analyzer_chain, 
        conversation_utterance_chain=conversation_utterance_chain,
        verbose=True
    )
    return rag_gpt_agent

def set_customer_details(agent, customer):
    agent.professional_name = f"{customer['First Name']} {customer['Last Name']}"
    agent.first_name = customer['First Name']
    agent.last_name = customer['Last Name']
    agent.gender = customer['Gender']
    agent.age = customer['Age']
    agent.region = customer['Region']
    agent.occupation = customer['Occupation']
    agent.policy_number = customer['Policy Number']
    agent.policy_start_date = customer['Policy Start Date']
    agent.policy_expiry_date = customer['Policy Expiry Date']
    agent.premium_type = customer['Premium Type']
    agent.product_type = customer['Product Type']
    agent.satisfaction_score = customer['Satisfaction Score']
    agent.late_payments = customer['Number of Late Payments']
    agent.preferred_communication = customer['Preferred Communication Channel']
    agent.customer_service_interactions = customer['Number of Customer Service Interactions']
    agent.claims_filed = customer['Number of Claims Filed']
    agent.total_claim_amount = customer['Total Claim Amount']
    agent.claim_frequency = customer['Claim Frequency']
    agent.credit_score = customer['Credit Score']
    agent.debt_to_income_ratio = customer['Debt-to-Income Ratio']

def main():
    rag_gpt_agent = initialize_agent()

    # Example customer data
    customer = {
        'First Name': 'John',
        'Last Name': 'Doe',
        'Gender': 'Male',
        'Age': 45,
        'Region': 'East',
        'Occupation': 'Engineer',
        'Policy Number': 'P123456',
        'Policy Start Date': '2019-01-01',
        'Policy Expiry Date': '2024-01-01',
        'Premium Type': 'Monthly',
        'Product Type': 'Life',
        'Satisfaction Score': 80,
        'Number of Late Payments': 1,
        'Preferred Communication Channel': 'Email',
        'Number of Customer Service Interactions': 2,
        'Number of Claims Filed': 0,
        'Total Claim Amount': 0.0,
        'Claim Frequency': 750.0,
        'Credit Score': 0.35,
        'Debt-to-Income Ratio': 0.25
    }

    set_customer_details(rag_gpt_agent, customer)
    rag_gpt_agent.seed_agent()

    print("Starting conversation with RAG-enabled GPT Agent. Type 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        rag_gpt_agent.human_step(user_input)
        bot_response = rag_gpt_agent.step()
        print(f"Sophia: {bot_response}")

if __name__ == "__main__":
    main()
