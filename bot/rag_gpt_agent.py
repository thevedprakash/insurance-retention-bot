from .gpt_agent import GPT, StageAnalyzerChain, ConversationChain
from pydantic import BaseModel, Field
from typing import Any, Dict, List
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

class RAGGPT(GPT):
    document_store: Dict[str, Any] = Field(default=None)
    retriever: Any = Field(default=None)

    def __init__(self, document_store: Dict[str, Any], retriever: Any, stage_analyzer_chain: StageAnalyzerChain, conversation_utterance_chain: ConversationChain, **kwargs):
        super().__init__(stage_analyzer_chain=stage_analyzer_chain, conversation_utterance_chain=conversation_utterance_chain, **kwargs)
        self.document_store = document_store
        self.retriever = retriever

    def step(self):
        # Retrieve relevant documents based on the conversation history for stage 6
        if self.current_conversation_stage == '6':
            context = " ".join(self.conversation_history)
            relevant_docs = self.retriever(context)
            relevant_text = " ".join([doc['text'] for doc in relevant_docs])

            # Add the relevant text to the prompt
            full_context = context + " " + relevant_text
        else:
            full_context = " ".join(self.conversation_history)

        ai_message = self.conversation_utterance_chain.run(
            person_name=self.person_name,
            person_role=self.person_role,
            team_name=self.team_name,
            conversation_purpose=self.conversation_purpose,
            conversation_history=full_context,
            conversation_stage=self.current_conversation_stage,
            conversation_type=self.conversation_type,
            professional_name=self.professional_name,
            first_name=self.first_name,
            last_name=self.last_name,
            gender=self.gender,
            age=self.age,
            region=self.region,
            occupation=self.occupation,
            policy_number=self.policy_number,
            policy_start_date=self.policy_start_date,
            policy_expiry_date=self.policy_expiry_date,
            premium_type=self.premium_type,
            product_type=self.product_type,
            satisfaction_score=self.satisfaction_score,
            late_payments=self.late_payments,
            preferred_communication=self.preferred_communication,
            customer_service_interactions=self.customer_service_interactions,
            claims_filed=self.claims_filed,
            total_claim_amount=self.total_claim_amount,
            claim_frequency=self.claim_frequency,
            credit_score=self.credit_score,
            debt_to_income_ratio=self.debt_to_income_ratio
        )
        self.conversation_history.append(ai_message)
        return ai_message.rstrip('<END_OF_TURN>')

    @classmethod
    def from_llm(cls, llm: ChatGoogleGenerativeAI, document_store: Dict[str, Any], retriever: Any, verbose: bool = False, **kwargs) -> "RAGGPT":
        stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)
        conversation_utterance_chain = ConversationChain.from_llm(llm, verbose=verbose)

        return cls(
            stage_analyzer_chain=stage_analyzer_chain,
            conversation_utterance_chain=conversation_utterance_chain,
            document_store=document_store,
            retriever=retriever,
            verbose=verbose,
            **kwargs,
        )
