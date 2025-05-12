"""
NeuroFlare AI-Powered Chatbot - Website Integration
---------------------------------------------------
This module provides integration for the NeuroFlare chatbot with websites,
allowing it to be embedded in any website as a chat widget.
"""

import os
import sys
import json
import logging
import asyncio
import uuid
from typing import Dict, List, Any, Optional
from aiohttp import web
import socketio
import requests
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import chatbot core and document integration
from prototypes.ai_chatbot.chatbot_core import ChatbotEngine, Message, Conversation
from prototypes.ai_chatbot.document_integration import DocumentAwareChatbot
from prototypes.ai_chatbot.organization_integration import OrganizationalChatbot

# ... (rest of the file continues as per original)

