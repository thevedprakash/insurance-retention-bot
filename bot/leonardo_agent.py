import logging
import smtplib
from flask import session
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

# Initialize the Language Model (LLM)
def initialize_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model='gemini-1.0-pro', temperature=0.2, google_api_key=api_key)
    return llm

llm = initialize_llm()

def generate_summary(docs):
    # Custom prompt for generating a summary
    custom_prompt_template = """
    Your job is to create a concise and informative summary of the conversation that highlights key points and any actionable items needed to reduce the risk of policy churn.

    Here is the conversation history:
    {docs}

    Please provide a summary of the conversation including:
    1. The main topics discussed.
    2. The customer's concerns and questions.
    3. Sophia's responses and any information provided.
    4. Any actionable items or follow-ups required.

    Summarize the conversation in a clear and concise manner, ensuring that the supervisor can quickly understand the key points and necessary actions.
    """

    # Create a PromptTemplate
    custom_prompt = PromptTemplate.from_template(custom_prompt_template)

    # Create a summarization chain with the custom prompt
    map_chain = LLMChain(llm=llm, prompt=custom_prompt)

    # Run the summarization chain with the docs as input
    summary = map_chain.run(docs=docs)
    return summary

def generate_email_content(conversation_summary, customer_name, supervisor_name):
    email_template = f"""
    Subject: Summary of Conversation with {customer_name}

    Dear {supervisor_name},

    I hope this email finds you well. Below is a summary of the recent conversation between Sophia and {customer_name}.

    Summary of the Conversation:
    {conversation_summary}

    Please review the summary and let me know if there are any further actions needed.

    Best regards,
    Leonardo
    """
    return email_template

def send_email(to_email, email_content):
    # Retrieve the email configuration directly using os.getenv
    from_email = os.getenv('MAIL_USERNAME')
    password = os.getenv('MAIL_PASSWORD')
    mail_server = os.getenv('MAIL_SERVER', 'sandbox.smtp.mailtrap.io')
    mail_port = int(os.getenv('MAIL_PORT', 2525))  # Defaulting to 2525 if not specified
    use_tls = os.getenv('MAIL_USE_TLS', 'True').lower() in ['true', '1', 't', 'y', 'yes']

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = "Conversation Summary"

    msg.attach(MIMEText(email_content, 'plain'))

    try:
        server = smtplib.SMTP(mail_server, mail_port)
        if use_tls:
            server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        logging.info("Email sent successfully")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")


def send_summary_to_supervisor():
    if 'conversation_history' not in session:
        logging.error("No conversation history found in session.")
        return {"error": "No conversation history found in session"}

    try:
        conversation_history = session['conversation_history']
        customer = session['customers'][0]
        supervisor_email = "thevedprakash.in@gmail.com"  # Supervisor email

        docs = [{"text": entry['message']} for entry in conversation_history]
        conversation_summary = generate_summary(docs)
        logging.info(f"conversation_summary : {conversation_summary}")
        email_content = generate_email_content(conversation_summary, customer['First Name'], "Supervisor")
        logging.info(f"email_content : {email_content}")
        send_email(supervisor_email, email_content)

        return {"message": "Summary sent to supervisor"}
    except Exception as e:
        logging.error(f"Error in send_summary_to_supervisor: {e}")
        raise
