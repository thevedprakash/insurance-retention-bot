import os
import math
from config import Config
import faiss
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever

def load_customer_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def relevance_score_fn(score: float) -> float:
    """
    Computes a relevance score for embedding vectors based on the provided score.
    
    Args:
        score (float): The initial score from the vector similarity measure.
    
    Returns:
        float: Adjusted relevance score.
    """
    """Calculate relevance score for vector embeddings."""
    return 1.0 - score / math.sqrt(2)

def create_new_memory_retriever():
    """
    Creates and configures a new memory retriever using Azure OpenAI embeddings with a FAISS vector store.
    
    Returns:
        TimeWeightedVectorStoreRetriever: Configured retriever ready for use with embedded data.
    """
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    embeddings_size =  768
    index = faiss.IndexFlatL2(embeddings_size)
    vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {}, relevance_score_fn=relevance_score_fn)
    return TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=15) 


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
