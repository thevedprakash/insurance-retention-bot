import logging
import os
from flask import Flask, request, jsonify, session, render_template
from flask_session import Session
import pandas as pd
from bot.gpt_agent import GPT
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
    return GPT.from_llm(llm=llm, verbose=True)

gpt_agent = initialize_agent()

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
        # # Replace NaN values with None (which will be converted to null in JSON)
        # df = df.where(pd.notnull(df), None)
        session['customers'] = df.to_dict(orient='records')
        logging.debug(f"Customers loaded: {session['customers']}")
        if session['customers']:
            customer = session['customers'][0]
            gpt_agent.professional_name = f"{customer['First Name']} {customer['Last Name']}"
            gpt_agent.seed_agent()  # Start with the initial stage
            initial_bot_message = gpt_agent.step()
            logging.debug(f"Initial bot message: {initial_bot_message}")
            conversation_history = [{"speaker": "Agent", "message": initial_bot_message}]
            session['conversation_history'] = conversation_history

            # Create a response dictionary
            response_data = {
                "customer": f"{customer}",
                "conversation_history": conversation_history
            }

            logging.debug(f"Response data: {response_data}")
            return jsonify(response_data)
        else:
            logging.error("No customers found in the uploaded file.")
            return jsonify({"error": "No customers found"}), 400

@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    global gpt_agent
    if 'customers' not in session:
        logging.error("No customers found in session.")
        return jsonify({"error": "No customers found in session"}), 400

    customer = session['customers'][0]  # Get the first customer for the conversation
    logging.debug(f"Starting conversation with customer: {customer}")
    gpt_agent.professional_name = f"{customer['First Name']} {customer['Last Name']}"
    gpt_agent.seed_agent()  # Start with the initial stage

    # Generate the initial bot message
    bot_message = gpt_agent.step()
    logging.debug(f"Initial bot message: {bot_message}")
    conversation_history = [{"speaker": "Agent", "message": bot_message}]
    session['conversation_history'] = conversation_history

    return jsonify({"conversation_history": conversation_history})

@app.route('/user_response', methods=['POST'])
def user_response():
    global gpt_agent
    data = request.json
    user_message = data['message']
    logging.debug(f"User message: {user_message}")

    gpt_agent.human_step(user_message)
    bot_message = gpt_agent.step()
    logging.debug(f"Bot message: {bot_message}")
    
    if 'conversation_history' not in session:
        session['conversation_history'] = []

    session['conversation_history'].append({"speaker": "User", "message": user_message})
    session['conversation_history'].append({"speaker": "Agent", "message": bot_message})

    return jsonify({"bot_response": bot_message, "conversation_history": session['conversation_history']})

@app.route('/next_customer', methods=['POST'])
def next_customer():
    global gpt_agent
    if 'customers' not in session:
        logging.error("No customers found in session.")
        return jsonify({"error": "No customers found in session"}), 400

    # Remove the first customer and move to the next one
    session['customers'].pop(0)
    if not session['customers']:
        logging.info("No more customers.")
        return jsonify({"message": "No more customers"}), 200

    customer = session['customers'][0]
    logging.debug(f"Next customer: {customer}")
    gpt_agent.professional_name = f"{customer['First Name']} {customer['Last Name']}"
    gpt_agent.seed_agent()  # Reset for the new customer

    return jsonify({"customer": customer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
