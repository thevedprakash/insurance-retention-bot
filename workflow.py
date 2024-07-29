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

def main():
    rag_gpt_agent = initialize_agent()

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
