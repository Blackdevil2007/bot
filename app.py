from fastapi import FastAPI, Request
from chatbot_core import Chatbot

app = FastAPI()
bot = Chatbot()

@app.get("/")
def home():
    return {"message": "🤖 AI Automation Bot is running!"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("user_input", "")
    response = bot.get_response(user_input)
    return {"response": response}
