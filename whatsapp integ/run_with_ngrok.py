import os
from flask import Flask, request
from pyngrok import ngrok
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

load_dotenv()

from rag_pipeline import app  # This is your LangChain app

flask_app = Flask(__name__)  # Use a separate name for Flask app

@flask_app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    incoming_msg = request.values.get('Body', '').strip()

    if not incoming_msg:
        return "Empty message", 400

    inputs = {"question": incoming_msg}
    result = app.invoke(inputs)
    answer = result.get("answer", "Sorry, I couldn't process your question.")

    response = MessagingResponse()
    msg = response.message()
    msg.body(answer)
    return str(response)

if __name__ == "__main__":
    public_url = ngrok.connect(5000)
    print(f"\n🌐 Ngrok is live: {public_url}/whatsapp")
    print("📨 Go to: https://www.twilio.com/console/sms/whatsapp/sandbox")
    print("Paste that URL in 'WHEN A MESSAGE COMES IN'")

    flask_app.run()
