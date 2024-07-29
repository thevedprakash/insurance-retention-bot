import logging
import os
from flask import Flask, request, jsonify, session, render_template
from flask_session import Session
import pandas as pd
from bot.gpt_agent import GPT, StageAnalyzerChain, ConversationChain
from bot.rag_gpt_agent import RAGGPT
from bot.document_processor import initialize_document_store, retrieve_relevant_documents
from config import Config
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from dotenv import load_dotenv

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load configuration
class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the GPT agent
def initialize_agent():
    api_key = Config.GOOGLE_API_KEY
    llm = ChatGoogleGenerativeAI(model='gemini-1.0-pro', temperature=0.2, google_api_key=api_key)
    return GPT.from_llm(llm=llm, verbose=True), StageAnalyzerChain.from_llm(llm=llm, verbose=True), ConversationChain.from_llm(llm=llm, verbose=True)

gpt_agent, stage_analyzer_chain, conversation_utterance_chain = initialize_agent()

# Initialize the document store and GPT agent
pdf_path = 'data/Principal-Sample-Life-Insurance-Policy.pdf'
document_store = initialize_document_store(pdf_path)

# Ensure document store is a list of dictionaries with a 'text' field
for doc in document_store:
    if 'text' not in doc:
        doc['text'] = "Default text if missing"

# Initialize the RAG-enabled GPT agent
rag_gpt_agent = RAGGPT(
    document_store=document_store, 
    retriever=lambda context: retrieve_relevant_documents(context, document_store=document_store, top_n=1), 
    stage_analyzer_chain=stage_analyzer_chain, 
    conversation_utterance_chain=conversation_utterance_chain
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    logging.info("File upload endpoint called.")
    if 'file' not in request.files:
        logging.error("No file part in the request.")
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        logging.error("No file selected for uploading.")
        return jsonify({"error": "No selected file"}), 400
    if file:
        logging.info("File successfully uploaded.")
        df = pd.read_csv(file)
        logging.info(f"Raw data loaded: {df}")

        # Handling missing or null values by replacing with appropriate defaults
        df.fillna({
            'First Name': 'Unknown',
            'Last Name': 'Unknown',
            'Gender': 'Unknown',
            'Age': 0,
            'Region': 'Unknown',
            'Occupation': 'Unknown',
            'Policy Number': 'Unknown',
            'Policy Start Date': 'Unknown',
            'Policy Expiry Date': 'Unknown',
            'Premium Type': 'Unknown',
            'Product Type': 'Unknown',
            'Satisfaction Score': 0,
            'Number of Late Payments': 0,
            'Preferred Communication Channel': 'Unknown',
            'Number of Customer Service Interactions': 0,
            'Number of Claims Filed': 0,
            'Total Claim Amount': 0.0,
            'Claim Frequency': 0.0,
            'Credit Score': 0.0,
            'Debt-to-Income Ratio': 0.0
        }, inplace=True)

        logging.info(f"After filling for any missing values: {df}")

        session['customers'] = df.to_dict(orient='records')
        logging.debug(f"Customers loaded: {session['customers']}")
        if session['customers']:
            customer = session['customers'][0]
            rag_gpt_agent.professional_name = f"{customer['First Name']} {customer['Last Name']}"
            rag_gpt_agent.first_name = customer['First Name']
            rag_gpt_agent.last_name = customer['Last Name']
            rag_gpt_agent.gender = customer['Gender']
            rag_gpt_agent.age = customer['Age']
            rag_gpt_agent.region = customer['Region']
            rag_gpt_agent.occupation = customer['Occupation']
            rag_gpt_agent.policy_number = customer['Policy Number']
            rag_gpt_agent.policy_start_date = customer['Policy Start Date']
            rag_gpt_agent.policy_expiry_date = customer['Policy Expiry Date']
            rag_gpt_agent.premium_type = customer['Premium Type']
            rag_gpt_agent.product_type = customer['Product Type']
            rag_gpt_agent.satisfaction_score = customer['Satisfaction Score']
            rag_gpt_agent.late_payments = customer['Number of Late Payments']
            rag_gpt_agent.preferred_communication = customer['Preferred Communication Channel']
            rag_gpt_agent.customer_service_interactions = customer['Number of Customer Service Interactions']
            rag_gpt_agent.claims_filed = customer['Number of Claims Filed']
            rag_gpt_agent.total_claim_amount = customer['Total Claim Amount']
            rag_gpt_agent.claim_frequency = customer['Claim Frequency']
            rag_gpt_agent.credit_score = customer['Credit Score']
            rag_gpt_agent.debt_to_income_ratio = customer['Debt-to-Income Ratio']

            rag_gpt_agent.seed_agent()  # Start with the initial stage
            initial_bot_message = rag_gpt_agent.step()
            logging.debug(f"Initial bot message: {initial_bot_message}")
            conversation_history = [{"speaker": "Agent", "message": initial_bot_message}]
            session['conversation_history'] = conversation_history

            # Create a response dictionary
            response_data = {
                "customer": customer,
                "conversation_history": conversation_history
            }

            logging.debug(f"Response data: {response_data}")
            return jsonify(response_data)
        else:
            logging.error("No customers found in the uploaded file.")
            return jsonify({"error": "No customers found"}), 400

@app.route('/user_response', methods=['POST'])
def user_response():
    global rag_gpt_agent
    data = request.json
    user_message = data['message']
    logging.debug(f"User message: {user_message}")

    rag_gpt_agent.human_step(user_message)
    bot_message = rag_gpt_agent.step()
    logging.debug(f"Bot message: {bot_message}")
    
    if 'conversation_history' not in session:
        session['conversation_history'] = []

    session['conversation_history'].append({"speaker": "User", "message": user_message})
    session['conversation_history'].append({"speaker": "Agent", "message": bot_message})

    return jsonify({"bot_response": bot_message, "conversation_history": session['conversation_history']})

@app.route('/next_customer', methods=['POST'])
def next_customer():
    global rag_gpt_agent
    if 'customers' not in session:
        logging.error("No customers found in session.")
        return jsonify({"error": "No customers found in session"}), 400

    # Get the current customer's info for summary
    current_customer = session['customers'][0]
    conversation_history = session.get('conversation_history', [])

    # Remove the first customer and move to the next one
    session['customers'].pop(0)
    if not session['customers']:
        logging.info("No more customers.")
        return jsonify({"message": "No more customers"}), 200

    customer = session['customers'][0]
    logging.debug(f"Next customer: {customer}")
    rag_gpt_agent.professional_name = f"{customer['First Name']} {customer['Last Name']}"
    rag_gpt_agent.first_name = customer['First Name']
    rag_gpt_agent.last_name = customer['Last Name']
    rag_gpt_agent.gender = customer['Gender']
    rag_gpt_agent.age = customer['Age']
    rag_gpt_agent.region = customer['Region']
    rag_gpt_agent.occupation = customer['Occupation']
    rag_gpt_agent.policy_number = customer['Policy Number']
    rag_gpt_agent.policy_start_date = customer['Policy Start Date']
    rag_gpt_agent.policy_expiry_date = customer['Policy Expiry Date']
    rag_gpt_agent.premium_type = customer['Premium Type']
    rag_gpt_agent.product_type = customer['Product Type']
    rag_gpt_agent.satisfaction_score = customer['Satisfaction Score']
    rag_gpt_agent.late_payments = customer['Number of Late Payments']
    rag_gpt_agent.preferred_communication = customer['Preferred Communication Channel']
    rag_gpt_agent.customer_service_interactions = customer['Number of Customer Service Interactions']
    rag_gpt_agent.claims_filed = customer['Number of Claims Filed']
    rag_gpt_agent.total_claim_amount = customer['Total Claim Amount']
    rag_gpt_agent.claim_frequency = customer['Claim Frequency']
    rag_gpt_agent.credit_score = customer['Credit Score']
    rag_gpt_agent.debt_to_income_ratio = customer['Debt-to-Income Ratio']

    rag_gpt_agent.seed_agent()  # Reset for the new customer

    return jsonify({"customer": customer})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
