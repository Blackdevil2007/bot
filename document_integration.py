"""
NeuroFlare AI-Powered Chatbot - Document Management Integration
--------------------------------------------------------------
This module demonstrates how the chatbot integrates with the NeuroFlare Document
Management System to provide document-related capabilities in chat conversations.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any, Tuple
import asyncio

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import chatbot core
from chatbot_core import ChatbotEngine, Intent, Conversation, Message

# Import document management components
try:
    from src.document_management.document_manager import DocumentManager, Document, DocumentType
    from src.document_management.search_engine import SearchEngine
    DOCUMENT_MANAGER_AVAILABLE = True
except ImportError:
    logging.warning("Document management modules not found. Using mock implementation.")
    DOCUMENT_MANAGER_AVAILABLE = False

# ... (rest of the file continues as per original)

