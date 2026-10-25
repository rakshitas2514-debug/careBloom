# CareBloom — Groq AI Edition

CareBloom is a Streamlit wellness and fitness companion. This copy is configured to use a **Groq API key (`gsk_...`)**, not a Gemini key.

## Important: AI availability

The project retries some temporary connection/server failures and shows an offline wellness fallback if Groq cannot be reached. No application can guarantee that a third-party AI API is available 24/7: service outages, internet issues, API quotas/rate limits, account restrictions, or model changes can interrupt AI replies.

## 1. Set up your Groq key (Windows)

1. Open the `CareBloom_Groq_Ready_Project` folder.
2. Open the `.streamlit` folder.
3. Copy `secrets.toml.example` and rename the copy to `secrets.toml`.
4. Open `secrets.toml` in Notepad.
5. Replace `PASTE_YOUR_GSK_KEY_HERE` with your own Groq API key. Keep the quotes.
6. Save the file.

Alternatively, create a file named `.env` in the project root (same folder as `app.py`) containing:

    GROQ_API_KEY=gsk_your_actual_key_here
    GROQ_MODEL=llama-3.3-70b-versatile

Use either method. Never share your key or commit `.env` / `secrets.toml` to GitHub. These secret files are excluded by `.gitignore`.

## 2. Run CareBloom

1. Install Python 3.10 or newer if it is not installed.
2. Double-click `run_carebloom.bat`. It installs the packages listed in `requirements.txt` and starts Streamlit.
3. Open the local URL printed in the terminal, usually `http://localhost:8501`.
4. Keep the terminal window open while using the site.

You can also open Command Prompt in this folder and run:

    python -m pip install -r requirements.txt
    python -m streamlit run app.py

## 3. Test the Groq connection

With your key configured, run:

    python test_groq_connection.py

This test does not print your API key. If it fails, use the HTTP status it reports:
- **401**: key is invalid or copied incorrectly.
- **403**: account/model access is restricted.
- **404**: model ID is unavailable; check the current model list in Groq Console.
- **429**: rate limit or quota reached.
- Connection error: check internet access, firewall, or temporary service availability.

## 4. Deploying later

For Streamlit Community Cloud, deploy this folder/repository and add `GROQ_API_KEY` (and optionally `GROQ_MODEL`) in the app's Secrets settings. Do not upload your real key to GitHub.

## Health note

CareBloom provides general wellness information only. It does not diagnose conditions, prescribe medication, or replace a qualified health professional.
