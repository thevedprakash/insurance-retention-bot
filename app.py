import logging
import os
from flask import Flask, request, jsonify, session, render_template
from flask_session import Session
import pandas as pd
from bot.gpt_agent import GPT, StageAnalyzerChain, ConversationChain
from bot.rag_gpt_agent import RAGGPT
from bot.utils import set_customer_details
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
    # llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest', temperature=0.2, google_api_key=api_key)
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

        session['customers'] = df.to_dict(orient='records')
        logging.debug(f"Customers loaded: {session['customers']}")
        if session['customers']:
            customer = session['customers'][0]
            set_customer_details(rag_gpt_agent, customer)
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

    end_conversation_phrases = ["bye", "goodbye", "i'm good", "no i don't need further assistance", "i'm done", "that's all", "talk later"]

    # Detect if the user is ending the conversation
    if any(phrase in user_message.lower() for phrase in end_conversation_phrases):
        logging.info("End conversation detected.")
        bot_message = "Okay, I'll make a note of that. Goodbye! 😊"
        session['conversation_history'].append({"speaker": "User", "message": user_message})
        session['conversation_history'].append({"speaker": "Agent", "message": bot_message})

        # Move to the next customer
        response = next_customer()
        return response

    # Continue the conversation if not ending
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
    set_customer_details(rag_gpt_agent, customer)
    rag_gpt_agent.seed_agent()  # Reset for the new customer

    initial_bot_message = rag_gpt_agent.step()
    logging.debug(f"Initial bot message for next customer: {initial_bot_message}")
    conversation_history = [{"speaker": "Agent", "message": initial_bot_message}]
    session['conversation_history'] = conversation_history

    return jsonify({"message": "Moved to the next customer.", "customer": customer, "conversation_history": conversation_history})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
