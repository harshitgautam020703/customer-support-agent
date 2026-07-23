"""
Main entry point for Customer Support Agent
"""

import argparse
import sys
from src.agent import CustomerSupportAgent
from src.utils import logger

def interactive_mode():
    """Run in interactive mode"""
    agent = CustomerSupportAgent()
    
    print("\n" + "="*60)
    print("🤖 Customer Support Agent (CloudSync Pro)")
    print("="*60)
    print("Type 'quit' to exit")
    print("Type 'history' to see your conversation history")
    print("Type 'reset' to reset the conversation")
    print("="*60 + "\n")
    
    user_id = input("Please enter your user ID (or press Enter for anonymous): ").strip()
    if not user_id:
        user_id = "anonymous"
    
    print(f"\nWelcome {user_id}! How can I help you today?\n")
    
    while True:
        query = input("You: ").strip()
        
        if query.lower() == 'quit':
            print("\nThank you for using CloudSync Pro Support. Have a great day! 👋")
            break
            
        elif query.lower() == 'history':
            history = agent.get_conversation_history(user_id)
            if history:
                print("\n📋 Your Conversation History:")
                for i, conv in enumerate(history[:5], 1):
                    print(f"\n{i}. You: {conv['user_input']}")
                    print(f"   Agent: {conv['response'][:100]}...")
            else:
                print("\nNo conversation history found.")
            continue
            
        elif query.lower() == 'reset':
            print("\n🔄 Resetting conversation...")
            continue
            
        if not query:
            continue
        
        # Process query
        result = agent.process_query(query, user_id)
        
        if result['success']:
            escalation_indicator = " [ESCALATED]" if result.get('escalated') else ""
            print(f"\nAgent{escalation_indicator}: {result['response']}\n")
            
            if result.get('escalated'):
                print(f"🔴 Case escalated. Reason: {result.get('escalation_reason')}")
        else:
            print(f"\n❌ Error: {result.get('error', 'Unknown error')}\n")

def web_mode():
    """Run in web mode"""
    import subprocess
    print("🚀 Starting FastAPI server...")
    subprocess.run(["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

def main():
    parser = argparse.ArgumentParser(description="Customer Support Agent")
    parser.add_argument(
        "--mode", 
        choices=["interactive", "web"],
        default="interactive",
        help="Run mode: interactive CLI or web API"
    )
    parser.add_argument(
        "--kb-dir",
        help="Custom knowledge base directory"
    )
    args = parser.parse_args()
    
    if args.mode == "web":
        web_mode()
    else:
        interactive_mode()

if __name__ == "__main__":
    main()