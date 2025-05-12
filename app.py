"""
NeuroFlare AI-Powered Chatbot - WhatsApp Integration
----------------------------------------------------
This module enables the NeuroFlare chatbot to be integrated with WhatsApp
using the WhatsApp Business API.
"""

import os
import sys
import json
import logging
import asyncio
import requests
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import chatbot core and document integration
from prototypes.ai_chatbot.chatbot_core import ChatbotEngine, Message, Conversation
from prototypes.ai_chatbot.document_integration import DocumentAwareChatbot

# ... (rest of the file continues as per original)

