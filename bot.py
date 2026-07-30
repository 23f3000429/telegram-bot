import os
import io
import sys
import json
import time
import asyncio
import threading
import urllib.request
import traceback
from contextlib import redirect_stdout, redirect_stderr

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

# --- Environment Variables & Config ---
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
AIPIPE_TOKEN = os.environ["AIPIPE_TOKEN"]

# Automatically construct the Render host URL for log_url
BASE_URL = os.environ.get("BASE_URL", "").rstrip("/")
if not BASE_URL:
    # Fallback to RENDER_EXTERNAL_URL if set by Render automatically
    BASE_URL = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:10000").rstrip("/")

LOG_URL = f"{BASE_URL}/run.jsonl"
LOG_FILE = "run.jsonl"

conversation_history = {}

client = OpenAI(base_url="https://aipipe.org/openai/v1", api_key=AIPIPE_TOKEN)

# Ensure log file exists locally
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        pass

# --- 1. Logging & Execution Tools ---
def log_event(event: dict):
    event["timestamp"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")

def run_python(code: str) -> str:
    """Executes Python code to download and compute dataset results."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    local_scope = {}
    
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(code, globals(), local_scope)
        output = stdout_buf.getvalue() + stderr_buf.getvalue()
    except Exception:
        output = f"Execution Error:\n{traceback.format_exc()}"
    
    return output[-8000:] if output else "Executed with no output."

tools = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run Python code to fetch and analyze datasets using pandas, requests, bs4, openpyxl, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code block to execute"}
                },
                "required": ["code"]
            }
        }
    }
]

# --- 2. Agent Execution Loop ---
async def agent_loop(chat_id: int, user_text: str) -> str:
    start_time = time.time()
    DEADLINE = 210  # 3.5 min deadline (budget under 300s timeout)

    history = conversation_history.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})

    system_prompt = (
        "You are a careful data analyst. The user asks a data-analysis question "
        "and gives the exact JSON output format required.\n"
        "RULES:\n"
        "1. Work out the answer using run_python to fetch/compute data when necessary. Never guess numbers.\n"
        "2. If python fails, answer from your knowledge.\n"
        "3. Reply with ONLY the exact JSON object requested. No markdown fences, no prose.\n"
        "4. Set \"log_url\": \"\" as a placeholder in your JSON.\n"
        "5. If a message is just setup ('I will send data next'), reply with a small JSON ack like {\"status\": \"ok\", \"log_url\": \"\"}.\n"
        "6. Match keys, nesting, and value types exactly as requested."
    )

    messages = [{"role": "system", "content": system_prompt}] + history[-20:]

    for step in range(10):
        elapsed = time.time() - start_time
        use_tools = tools if elapsed < DEADLINE else None

        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # Frontier model required by grading pipeline
                messages=messages,
                tools=use_tools,
                tool_choice="auto" if use_tools else "none"
            )
        except Exception as e:
            log_event({"type": "llm_error", "chat_id": chat_id, "error": str(e)})
            return json.dumps({"answer": "error processing request", "log_url": LOG_URL})

        msg = response.choices[0].message

        if msg.tool_calls and elapsed < DEADLINE:
            messages.append(msg)
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "run_python":
                    args = json.loads(tool_call.function.arguments)
                    code = args.get("code", "")
                    
                    log_event({"type": "tool_call", "chat_id": chat_id, "code": code})
                    result = run_python(code)
                    log_event({"type": "tool_result", "chat_id": chat_id, "output": result})

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
        else:
            raw_reply = msg.content or ""
            history.append({"role": "assistant", "content": raw_reply})
            return raw_reply

    return history[-1]["content"] if history else "{}"

def clean_and_format_json(raw_text: str) -> str:
    """Strips markdown fences and ensures clean JSON with proper log_url."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(text[start:end+1])
            except Exception:
                parsed = {"answer": text}
        else:
            parsed = {"answer": text}

    if not isinstance(parsed, dict):
        parsed = {"answer": parsed}

    parsed["log_url"] = LOG_URL
    return json.dumps(parsed)

# --- 3. Telegram Handler ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text
    log_event({"type": "incoming", "chat_id": chat_id, "text": user_text})

    try:
        raw_response = await agent_loop(chat_id, user_text)
        final_reply = clean_and_format_json(raw_response)
    except Exception as e:
        log_event({"type": "handler_error", "chat_id": chat_id, "error": str(e)})
        final_reply = json.dumps({"answer": "internal error", "log_url": LOG_URL})

    log_event({"type": "outgoing", "chat_id": chat_id, "text": final_reply})
    await update.message.reply_text(final_reply)

# --- 4. FastAPI Web Server ---
app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True, "time": time.time()}

@app.get("/run.jsonl")
def serve_logs():
    if os.path.exists(LOG_FILE):
        return FileResponse(LOG_FILE, media_type="application/x-ndjson")
    return JSONResponse(status_code=404, content={"error": "Log file not found"})

def self_ping():
    while True:
        time.sleep(600)  # Ping every 10 min to keep Render awake
        try:
            req = urllib.request.Request(f"{BASE_URL}/health", headers={"User-Agent": "KeepAlive/1.0"})
            with urllib.request.urlopen(req) as resp:
                _ = resp.read()
        except Exception:
            pass

# --- 5. Application Launcher ---
def start_bot():
    """Runs the telegram polling engine inside a dedicated event loop."""
    async def _run():
        await tb_app.initialize()
        await tb_app.start()
        await tb_app.updater.start_polling(drop_pending_updates=True)
        print("Telegram polling active...")
        # Keeps the polling loop alive endlessly in this background thread
        while True:
            await asyncio.sleep(3600)

    # Create and run a new asyncio loop dedicated to this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_run())

if __name__ == "__main__":
    # 1. Start Keep-Alive Ping in Background Thread
    threading.Thread(target=self_ping, daemon=True).start()
    
    # 2. Start Telegram Polling in Background Thread (fixed non-blocking engine)
    threading.Thread(target=start_bot, daemon=True).start()

    # 3. Start FastAPI Web Server in Main Thread
    port = int(os.environ.get("PORT", 10000))
    print(f"FastAPI running on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
