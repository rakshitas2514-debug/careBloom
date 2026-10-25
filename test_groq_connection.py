"""Simple Groq connection test. Never prints the API key."""
import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

key = os.environ.get("GROQ_API_KEY", "").strip()
if not key:
    try:
        import streamlit as st
        key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except Exception:
        key = ""

if not key or key == "gsk_paste_your_own_key_here":
    print("GROQ_API_KEY is missing. Add it to .env or .streamlit/secrets.toml.")
    raise SystemExit(1)

payload = {    
    "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
    "messages": [{"role": "user", "content": "Reply with: CareBloom connection successful."}],
    "max_completion_tokens": 40,
    "temperature": 0
}
request = urllib.request.Request(
    "https://api.groq.com/openai/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={
    "Content-Type": "application/json",
    "Authorization": f"Bearer {key}",
    "User-Agent": "CareBloom/1.0"
    }, 
    method="POST"
)
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    print("SUCCESS: Groq API responded.")
    print(data["choices"][0]["message"]["content"])
except urllib.error.HTTPError as exc:
    print(f"Groq returned HTTP {exc.code}.")
    try:
        error_body = exc.read().decode("utf-8", errors="replace")
        print("Groq error details:")
        print(error_body)
    except Exception:
        print("Could not read the error details.")
    raise SystemExit(1)
except Exception as exc:
    print(f"Connection test failed: {type(exc).__name__}. Check your internet connection.")
    raise SystemExit(1)
