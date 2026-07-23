"""
Customer Support Agent with RAG and LangGraph
"""

import os
import re
import json
import hashlib
from typing import Annotated, Literal, TypedDict, List, Optional
from datetime import datetime
import uuid

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint import MemorySaver

from src.knowledge_base import KnowledgeBaseManager
from src.escalation import EscalationManager
from src.memory import ConversationMemory
from src.utils import logger, load_config, format_response

load_dotenv()

class SupportState(TypedDict):
    """State schema for the support agent"""
    messages: Annotated[list, add_messages]
    user_input: str
    user_id: str
    session_id: str
    retrieved_context: str
    response: str
    escalate: bool
    escalation_reason: str
    sentiment_score: float
    intent: str
    confidence_score: float

class CustomerSupportAgent:
    """Main customer support agent with RAG capabilities"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.kb_manager = KnowledgeBaseManager(self.config)
        self.escalation_manager = EscalationManager(self.config)
        self.memory = ConversationMemory()
        self.vectorstore = None
        self.llm = None
        self.embeddings = None
        self.graph = None
        
        # Initialize components
        self._initialize_llm()
        self._initialize_embeddings()
        self._build_knowledge_base()
        self._build_graph()
        
    def _initialize_llm(self):
        """Initialize LLM"""
        self.llm = ChatOpenAI(
            model=self.config.get('llm.model', 'gpt-4o-mini'),
            temperature=self.config.get('llm.temperature', 0.2),
            max_tokens=self.config.get('llm.max_tokens', 1000)
        )
        logger.info("✅ LLM initialized")
        
    def _initialize_embeddings(self):
        """Initialize embeddings"""
        self.embeddings = OpenAIEmbeddings(
            model=self.config.get('embeddings.model', 'text-embedding-ada-002')
        )
        logger.info("✅ Embeddings initialized")
        
    def _build_knowledge_base(self):
        """Build or load knowledge base"""
        if os.path.exists(self.config.get('vectorstore.path', 'data/vectorstore')):
            # Load existing vectorstore
            self.vectorstore = FAISS.load_local(
                self.config.get('vectorstore.path', 'data/vectorstore'),
                self.embeddings
            )
            logger.info("✅ Loaded existing vectorstore")
        else:
            # Build new vectorstore
            documents = self.kb_manager.load_documents()
            self.vectorstore = self.kb_manager.build_vectorstore(documents)
            logger.info(f"✅ Built new vectorstore with {len(documents)} documents")
            
    def _build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(SupportState)
        
        # Add nodes
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("analyze_sentiment", self.analyze_sentiment)
        workflow.add_node("check_escalation", self.check_escalation)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("save_conversation", self.save_conversation)
        
        # Add edges
        workflow.set_entry_point("retrieve_context")
        workflow.add_edge("retrieve_context", "analyze_sentiment")
        workflow.add_edge("analyze_sentiment", "check_escalation")
        
        # Conditional edge for escalation
        workflow.add_conditional_edges(
            "check_escalation",
            self.route_after_escalation,
            {
                "generate_response": "generate_response",
                "escalate": "generate_response"  # Both go to generate, but with different context
            }
        )
        
        workflow.add_edge("generate_response", "save_conversation")
        workflow.add_edge("save_conversation", END)
        
        # Compile with memory
        self.graph = workflow.compile(checkpointer=MemorySaver())
        logger.info("✅ Graph workflow compiled")
        
    def retrieve_context(self, state: SupportState) -> SupportState:
        """Retrieve relevant context from knowledge base"""
        query = state["user_input"]
        
        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search_with_score(query, k=3)
        
        # Filter by threshold
        threshold = self.config.get('retrieval.similarity_threshold', 0.7)
        filtered_docs = [(doc, score) for doc, score in docs if score >= threshold]
        
        if filtered_docs:
            context = "\n\n".join([doc.page_content for doc, _ in filtered_docs])
        else:
            # Fallback to general context
            context = self.kb_manager.get_fallback_context()
            
        return {"retrieved_context": context}
        
    def analyze_sentiment(self, state: SupportState) -> SupportState:
        """Analyze customer sentiment"""
        text = state["user_input"]
        
        # Simple sentiment analysis
        positive_words = ["great", "good", "excellent", "happy", "satisfied", "thank", "appreciate"]
        negative_words = ["bad", "terrible", "awful", "frustrated", "angry", "disappointed", "upset", "horrible"]
        
        pos_count = sum(1 for word in positive_words if word in text.lower())
        neg_count = sum(1 for word in negative_words if word in text.lower())
        
        if pos_count > neg_count:
            sentiment_score = 0.7
        elif neg_count > pos_count:
            sentiment_score = 0.3
        else:
            sentiment_score = 0.5
            
        # Update escalation threshold based on sentiment
        if sentiment_score < 0.4:
            self.escalation_manager.lower_threshold()
            
        return {"sentiment_score": sentiment_score}
        
    def check_escalation(self, state: SupportState) -> SupportState:
        """Check if query needs escalation"""
        text = state["user_input"]
        sentiment = state.get("sentiment_score", 0.5)
        
        # Check escalation rules
        should_escalate, reason = self.escalation_manager.check_escalation(
            text, sentiment
        )
        
        return {
            "escalate": should_escalate,
            "escalation_reason": reason
        }
        
    def generate_response(self, state: SupportState) -> SupportState:
        """Generate response using RAG"""
        should_escalate = state.get("escalate", False)
        user_input = state["user_input"]
        
        if should_escalate:
            # Generate escalation response
            response_text = self._generate_escalation_response(state)
            return {
                "response": response_text,
                "messages": [AIMessage(content=response_text)]
            }
        
        # Generate RAG response
        context = state.get("retrieved_context", "")
        conversation = state["messages"][:-1]  # Exclude latest user message
        
        # Prepare system prompt
        system_prompt = f"""You are a friendly and helpful customer support agent for CloudSync Pro.
        
Use this knowledge base context to answer accurately:
{context}

Guidelines:
- Be friendly, concise, and solution-focused
- If you don't know something, be honest and offer to escalate
- Use a conversational tone
- Provide step-by-step instructions when needed
- Acknowledge the customer's concern

Company Information:
- Product: CloudSync Pro
- Support Hours: 24/7
- Response Time: Usually within minutes
"""

        # Build messages
        messages = [
            SystemMessage(content=system_prompt),
            *conversation,
            HumanMessage(content=user_input)
        ]
        
        # Generate response
        response = self.llm.invoke(messages)
        response_text = response.content
        
        # Format response
        response_text = format_response(response_text)
        
        return {
            "response": response_text,
            "messages": [AIMessage(content=response_text)]
        }
        
    def _generate_escalation_response(self, state: SupportState) -> str:
        """Generate escalation response"""
        case_id = f"CS-{hashlib.md5(state['user_input'].encode()).hexdigest()[:8].upper()}"
        reason = state.get("escalation_reason", "Complex issue")
        
        escalation_template = """I completely understand your concern and I want to make sure this gets the proper attention it deserves.

🔴 **Case Escalated**: {case_id}
**Reason**: {reason}

I'm connecting you with our senior support specialist who can resolve this directly. They have access to advanced tools and can provide the most comprehensive solution.

📧 You'll receive an email confirmation within 15 minutes
📞 A specialist will reach out within 2 hours
⏰ Priority response time: 2-4 hours

In the meantime, please ensure you:
1. Keep your case ID handy: {case_id}
2. Check your email for follow-up instructions
3. Reply to any emails from our specialist

We take your concerns very seriously and will do everything possible to resolve this to your satisfaction.

Thank you for your patience and understanding.

**Your Case ID**: {case_id}
"""
        
        return escalation_template.format(case_id=case_id, reason=reason)
        
    def route_after_escalation(self, state: SupportState) -> Literal["generate_response", "escalate"]:
        """Route after escalation check"""
        if state.get("escalate", False):
            return "generate_response"
        return "generate_response"
        
    def save_conversation(self, state: SupportState) -> SupportState:
        """Save conversation to memory"""
        # Generate conversation summary
        summary = {
            'user_id': state.get('user_id', 'anonymous'),
            'session_id': state.get('session_id', str(uuid.uuid4())),
            'timestamp': datetime.now().isoformat(),
            'user_input': state['user_input'],
            'response': state['response'],
            'escalated': state.get('escalate', False),
            'sentiment': state.get('sentiment_score', 0.5)
        }
        
        # Save to memory
        self.memory.save_conversation(
            state.get('user_id', 'anonymous'),
            summary
        )
        
        return state
        
    def process_query(self, user_input: str, user_id: str = None) -> dict:
        """Process a customer query"""
        session_id = str(uuid.uuid4())
        
        # Create initial state
        state = {
            "messages": [],
            "user_input": user_input,
            "user_id": user_id or "anonymous",
            "session_id": session_id,
            "retrieved_context": "",
            "response": "",
            "escalate": False,
            "escalation_reason": "",
            "sentiment_score": 0.5,
            "intent": "",
            "confidence_score": 0.0
        }
        
        # Add user message
        state["messages"].append(HumanMessage(content=user_input))
        
        # Run the graph
        try:
            result = self.graph.invoke(state)
            
            return {
                "success": True,
                "response": result.get("response", ""),
                "escalated": result.get("escalate", False),
                "escalation_reason": result.get("escalation_reason", ""),
                "session_id": session_id,
                "sentiment": result.get("sentiment_score", 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I'm having trouble processing your request. Please try again or contact support directly."
            }

    def get_conversation_history(self, user_id: str) -> List[dict]:
        """Get conversation history for a user"""
        return self.memory.get_conversation_history(user_id)