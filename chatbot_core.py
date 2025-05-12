"""
NeuroFlare AI-Powered Chatbot Core Module
-----------------------------------------
This module implements the core functionality of the AI-powered chatbot, 
integrating with the Document Management System and other NeuroFlare tools.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Optional, Union, Any
import uuid
import asyncio
import re
import requests

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class GeminiLLM:
    """Gemini LLM integration for NeuroFlare chatbot."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        data = {
            "contents": [
                {"role": m["role"], "parts": [{"text": m["content"]}]} for m in messages
            ]
        }
        try:
            resp = requests.post(self.api_url, headers=headers, params=params, json=data, timeout=15)
            resp.raise_for_status()
            result = resp.json()
            if "candidates" in result and result["candidates"]:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            return "[Gemini did not return a response]"
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            return "[Error communicating with Gemini LLM]"


# For NLP/AI capabilities
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except ImportError:
    logging.warning("NLTK not installed. Some NLP features will be limited.")

# Placeholder for actual AI model imports
# In production, this would use a proper NLP/ML framework
# such as TensorFlow, PyTorch, or a cloud AI service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Intent:
    """Class representing a user intent identified by the chatbot."""
    
    def __init__(self, intent_type: str, confidence: float, entities: Dict[str, Any] = None):
        self.intent_type = intent_type  # e.g., "product_inquiry", "order_status", etc.
        self.confidence = confidence    # 0.0 to 1.0
        self.entities = entities or {}  # Extracted entities like product names, dates, etc.
    
    def __repr__(self):
        return f"Intent(type={self.intent_type}, confidence={self.confidence:.2f}, entities={self.entities})"


class Message:
    """Class representing a message in a chat conversation."""
    
    def __init__(self, 
                 text: str, 
                 sender_type: str,  # "user" or "bot"
                 sender_id: str,
                 timestamp: Optional[datetime.datetime] = None,
                 attachments: Optional[List[Dict]] = None,
                 metadata: Optional[Dict] = None):
        self.id = str(uuid.uuid4())
        self.text = text
        self.sender_type = sender_type
        self.sender_id = sender_id
        self.timestamp = timestamp or datetime.datetime.now()
        self.attachments = attachments or []
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary for storage or transmission."""
        return {
            "id": self.id,
            "text": self.text,
            "sender_type": self.sender_type,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp.isoformat(),
            "attachments": self.attachments,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        """Create a Message object from a dictionary."""
        timestamp = datetime.datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else None
        return cls(
            text=data["text"],
            sender_type=data["sender_type"],
            sender_id=data["sender_id"],
            timestamp=timestamp,
            attachments=data.get("attachments"),
            metadata=data.get("metadata")
        )


class Conversation:
    """Class representing a chat conversation between a user and the bot."""
    
    def __init__(self, 
                 user_id: str,
                 conversation_id: Optional[str] = None,
                 metadata: Optional[Dict] = None):
        self.id = conversation_id or str(uuid.uuid4())
        self.user_id = user_id
        self.messages: List[Message] = []
        self.created_at = datetime.datetime.now()
        self.updated_at = self.created_at
        self.metadata = metadata or {}
        self.active = True
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.datetime.now()
    
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get conversation messages, optionally limited to the most recent ones."""
        if limit and limit > 0:
            return self.messages[-limit:]
        return self.messages
    
    def to_dict(self) -> Dict:
        """Convert conversation to dictionary for storage or transmission."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "active": self.active
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Conversation':
        """Create a Conversation object from a dictionary."""
        conversation = cls(
            user_id=data["user_id"],
            conversation_id=data["id"],
            metadata=data.get("metadata")
        )
        # Convert ISO format strings back to datetime objects
        conversation.created_at = datetime.datetime.fromisoformat(data["created_at"])
        conversation.updated_at = datetime.datetime.fromisoformat(data["updated_at"])
        conversation.active = data.get("active", True)
        
        # Add messages
        for msg_data in data.get("messages", []):
            conversation.messages.append(Message.from_dict(msg_data))
        
        return conversation


class NLPEngine:
    """
    Natural Language Processing engine for the chatbot.
    This is a simplified version. In a production environment,
    this would connect to a proper NLP service or use a more
    sophisticated ML model.
    """
    
    def __init__(self):
        # Simple patterns for intent matching - would be ML-based in production
        self.intent_patterns = {
            "greeting": [
                r"hello", r"hi", r"hey", r"greetings", r"good morning", r"good afternoon", r"good evening"
            ],
            "farewell": [
                r"bye", r"goodbye", r"see you", r"talk to you later", r"have a good day"
            ],
            "help": [
                r"help", r"assist", r"support", r"how can you help", r"what can you do"
            ],
            "order_status": [
                r"order status", r"where is my order", r"track order", r"when will my order arrive"
            ],
            "product_inquiry": [
                r"product", r"item", r"do you have", r"is there", r"availability", r"in stock"
            ],
            "invoice_inquiry": [
                r"invoice", r"bill", r"payment", r"receipt", r"paid", r"charge"
            ],
            "feedback": [
                r"feedback", r"suggest", r"improve", r"review", r"rate"
            ],
            "document_request": [
                r"document", r"file", r"upload", r"download", r"share", r"attachment"
            ]
        }
    
    def analyze_text(self, text: str) -> Intent:
        """
        Analyze text to determine the user intent.
        In a real implementation, this would use an actual NLP model.
        """
        text = text.lower()
        
        # Simple intent matching based on regex patterns
        matched_intent = "unknown"
        max_confidence = 0.0
        entities = {}
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    if max_confidence < 0.8:
                        matched_intent = intent
                        max_confidence = 0.8
        
        # Example: Extract product name or order number (very basic)
        if matched_intent == "order_status":
            match = re.search(r"order\s*#?\s*(\w+)", text)
            if match:
                entities["order_number"] = match.group(1)
        elif matched_intent == "invoice_inquiry":
            match = re.search(r"invoice\s*#?\s*(\w+)", text)
            if match:
                entities["invoice_number"] = match.group(1)
        
        return Intent(matched_intent, max_confidence, entities)

# --- Main Chatbot Engine (Gemini integration for fallback/advanced intents) ---

class ChatbotEngine:
    """
    Main chatbot engine for NeuroFlare, now with Gemini fallback/intent support.
    """
    def __init__(self, gemini_api_key: Optional[str] = None, gemini_intents: Optional[set] = None):
        self.nlp_engine = NLPEngine()
        self.active_conversations: Dict[str, Conversation] = {}
        self.gemini = GeminiLLM(gemini_api_key) if gemini_api_key else None
        # Intents for which to use Gemini (add more as needed)
        self.gemini_intents = gemini_intents or {"unknown"}

    async def process_message(self, user_id: str, message_text: str, conversation_id: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict:
        """
        Process a user message and generate a response.
        Uses Gemini for fallback/advanced intents if configured.
        """
        # Get or create conversation
        if conversation_id and conversation_id in self.active_conversations:
            conversation = self.active_conversations[conversation_id]
        else:
            conversation = Conversation(user_id=user_id)
            self.active_conversations[conversation.id] = conversation

        # Add user message
        user_msg = Message(text=message_text, sender_type="user", sender_id=user_id, metadata=metadata)
        conversation.add_message(user_msg)

        # Detect intent
        intent = self.nlp_engine.analyze_text(message_text)

        # Decide response logic
        if self.gemini and (intent.intent_type in self.gemini_intents or intent.intent_type == "unknown"):
            # Use Gemini for this intent
            prompt = f"User: {message_text}\nRespond as a helpful business assistant."
            gemini_response = self.gemini.generate_response(prompt)
            bot_text = gemini_response
        else:
            # Rule-based/default responses
            bot_text = self._default_response(intent, message_text)

        # Add bot message
        bot_msg = Message(text=bot_text, sender_type="bot", sender_id="neuroflare-bot")
        conversation.add_message(bot_msg)

        return {
            "conversation_id": conversation.id,
            "message": bot_msg.to_dict(),
            "intent": intent.intent_type,
            "entities": intent.entities
        }

    def _default_response(self, intent: Intent, message_text: str) -> str:
        """
        Simple rule-based responses for known intents.
        """
        if intent.intent_type == "greeting":
            return "Hello! How can I assist you today?"
        if intent.intent_type == "farewell":
            return "Goodbye! If you have more questions, just ask."
        if intent.intent_type == "help":
            return "I'm here to help with your business queries, orders, documents, and more."
        if intent.intent_type == "order_status":
            order = intent.entities.get("order_number", "your order")
            return f"Let me check the status of {order}. Please wait a moment."
        if intent.intent_type == "product_inquiry":
            return "Sure, I can help with product information. What would you like to know?"
        if intent.intent_type == "invoice_inquiry":
            invoice = intent.entities.get("invoice_number", "your invoice")
            return f"Let me look up {invoice}. Please wait a moment."
        if intent.intent_type == "feedback":
            return "Thank you for your feedback! We appreciate your input."
        if intent.intent_type == "document_request":
            return "Can you specify which document you need help with?"
        return "I'm not sure how to help with that. Could you please rephrase your question?"


