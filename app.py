from fastapi import FastAPI
from chatbot_core import Chatbot

app = FastAPI()
bot = Chatbot()

@app.get("/")
def home():
    return {"message": "Bot is live!"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_input = data.get("user_input", "")
    return {"response": bot.get_response(user_input)}
