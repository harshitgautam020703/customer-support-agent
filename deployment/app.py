"""
FastAPI Application for Customer Support Agent
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import uvicorn
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.agent import CustomerSupportAgent

app = FastAPI(
    title="Customer Support Agent API",
    description="AI-powered customer support with RAG and escalation",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent
agent = CustomerSupportAgent()

# Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=5000)
    user_id: Optional[str] = None
    metadata: Optional[Dict] = None

class QueryResponse(BaseModel):
    success: bool
    response: str
    escalated: bool
    escalation_reason: Optional[str] = None
    session_id: str
    timestamp: str
    confidence: Optional[float] = None

class ConversationHistory(BaseModel):
    user_id: str
    conversations: List[Dict]
    total: int

class EscalationStats(BaseModel):
    total_escalations: int
    average_resolution_time: str
    escalation_rate: float

# Endpoints
@app.get("/")
async def root():
    return {
        "service": "Customer Support Agent",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": [
            "/query",
            "/history/{user_id}",
            "/escalations",
            "/health"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a customer query"""
    try:
        result = agent.process_query(request.query, request.user_id)
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Unknown error processing query')
            )
        
        return QueryResponse(
            success=True,
            response=result.get('response', ''),
            escalated=result.get('escalated', False),
            escalation_reason=result.get('escalation_reason', ''),
            session_id=result.get('session_id', ''),
            timestamp=datetime.now().isoformat(),
            confidence=result.get('confidence', 0.0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{user_id}", response_model=ConversationHistory)
async def get_history(user_id: str, limit: int = 10):
    """Get conversation history for a user"""
    try:
        history = agent.get_conversation_history(user_id)
        
        return ConversationHistory(
            user_id=user_id,
            conversations=history[:limit],
            total=len(history)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/escalations", response_model=EscalationStats)
async def get_escalation_stats():
    """Get escalation statistics"""
    try:
        stats = agent.escalation_manager.get_escalation_stats()
        return EscalationStats(**stats)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_agent():
    """Reset agent state (admin only)"""
    try:
        agent.escalation_manager.reset_threshold()
        return {"status": "reset", "message": "Agent state reset successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge/add")
async def add_knowledge(content: str, category: str = "general"):
    """Add new knowledge to the system (admin only)"""
    try:
        agent.kb_manager.add_document(
            content=content,
            metadata={'category': category, 'source': f'manual_{datetime.now().isoformat()}'}
        )
        return {"status": "success", "message": "Knowledge added successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )