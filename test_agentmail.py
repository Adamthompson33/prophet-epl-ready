"""
AgentMail test script for Prophet EPL.
Run: python3 test_agentmail.py
Requires: pip install agentmail python-dotenv
"""
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env

api_key = os.getenv("AGENTMAIL_KEY")
if not api_key:
    print("❌ AGENTMAIL_KEY not set")
    exit(1)

print(f"✓ AGENTMAIL_KEY loaded: {api_key[:20]}...")

# Test AgentMail SDK
try:
    from agentmail import AgentMail
    client = AgentMail(api_key=api_key)
    print("✓ AgentMail client initialized")
    
    # List inboxes (if any exist)
    inboxes = client.inboxes.list()
    print(f"✓ Found {len(inboxes)} inbox(es)")
    
except ImportError:
    print("❌ agentmail not installed. Run: pip3 install agentmail python-dotenv")
except Exception as e:
    print(f"❌ Error: {e}")
