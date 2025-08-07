from flask import Flask, request
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import threading
import os
from dotenv import load_dotenv
from rag_pipeline import app as rag_app  # LangChain graph

load_dotenv()

app = Flask(__name__)
client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

def process_question(user_text, user_id, channel_id):
    # Notify user
    try:
        client.chat_postMessage(
            channel=channel_id,
            text=f"<@{user_id}> asked: *{user_text}*\nWorking on it... :hourglass_flowing_sand:"
        )
    except SlackApiError as e:
        print(f"Slack error (initial msg): {e.response['error']}")

    # LangChain processing
    try:
        result = rag_app.invoke({"question": user_text})
        final_answer = result.get("answer", "🤖 No response generated.")
    except Exception as e:
        final_answer = f"❌ Error from LangChain: {str(e)}"

    # Send the answer
    try:
        client.chat_postMessage(
            channel=channel_id,
            text=f":brain: *Answer:* {final_answer}"
        )
    except SlackApiError as e:
        print(f"Slack error (final msg): {e.response['error']}")

@app.route("/slack/command", methods=["POST"])
def slash_command():
    data = request.form
    user_text = data.get("text", "")
    user_id = data.get("user_id")
    channel_id = data.get("channel_id")

    # Process in background
    thread = threading.Thread(
        target=process_question,
        args=(user_text, user_id, channel_id)
    )
    thread.start()

    # Immediate response to Slack
    return "", 200

if __name__ == "__main__":
    app.run(port=5000)
