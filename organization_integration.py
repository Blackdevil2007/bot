"""
NeuroFlare AI-Powered Chatbot - Organizational Integration
----------------------------------------------------------
This module connects the chatbot with internal organizational systems such as:
- Microsoft Teams/Slack for workplace messaging
- Internal knowledge bases and document repositories
- Employee directories and organizational structures
- ERP, CRM, and other business systems
"""

import os
import sys
import json
import logging
import asyncio
import requests
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import chatbot core and document integration
from prototypes.ai_chatbot.chatbot_core import ChatbotEngine, Message, Conversation, Intent
from prototypes.ai_chatbot.document_integration import DocumentAwareChatbot

# ... (rest of the file continues as per original)

