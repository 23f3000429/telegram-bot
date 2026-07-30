import os
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

# --- Environment Variables ---
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
AIPIPE_TOKEN = os.environ["AIPIPE_TOKEN"]
LOG_URL = os.environ.get(
    "LOG_URL",
    "https://raw.githubusercontent.com/23f3000429/telegram-bot/main/run.jsonl"
)

LOG_FILE = "run.jsonl"
conversation_history = {}

client = OpenAI(base_url="https://aipipe.org/openai/v1", api_key=AIPIPE_TOKEN)

# --- 1. Background HTTP Server for Render Health Checks ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        return  # Silence standard HTTP access logs

def start_http_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    print(f"HTTP health-check server running on port {port}")
    server.serve_forever()

# --- 2. Logging & Message Processing Logic ---
def log_event(event: dict):
    event["timestamp"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Guard against non-text messages (e.g. photos, stickers, system events)
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text
    log_event({"type": "incoming", "chat_id": chat_id, "text": user_text})

    history = conversation_history.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})

    system_prompt = (
        "You are a careful data analyst. The user's LAST message asks a data-analysis "
        "question and tells you exactly what JSON shape to reply with. Work out the "
        "real answer (use any public data you know, e.g. MOSPI statistics, general "
        "world knowledge, or arithmetic on numbers given in the message). "
        "Reply with ONLY that exact JSON object and absolutely nothing else — no "
        "explanation, no markdown, no code fences, just the raw JSON."
    )
    
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "system", "content": system_prompt}] + history[-6:],
    )
    reply_text = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": reply_text})

    try:
        parsed = json.loads(reply_text)
    except json.JSONDecodeError:
        start, end = reply_text.find("{"), reply_text.rfind("}")
        parsed = json.loads(reply_text[start:end + 1])
        
    parsed["log_url"] = LOG_URL
    final_reply = json.dumps(parsed)

    log_event({"type": "outgoing", "chat_id": chat_id, "text": final_reply})
    await update.message.reply_text(final_reply)

# --- 3. Program Entrypoint ---
if __name__ == "__main__":
    # Start the HTTP server in a separate background thread
    threading.Thread(target=start_http_server, daemon=True).start()

    # Build and run the Telegram bot polling process
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot is running...")
    app.run_polling()