import os, json, sqlite3, hashlib, hmac, re, urllib.request, urllib.error, http.client, socket, time, base64
from pathlib import Path
from datetime import date
from urllib.parse import quote
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "carebloom.db"
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b").strip() or "llama-3.3-70b-versatile"
st.set_page_config(page_title="CareBloom | Health & Fitness Companion", page_icon="🌸", layout="wide")

# Keep Streamlit login/register text fields readable on light backgrounds.
st.markdown("""
<style>
div[data-testid="stTextInput"] input {
    background-color: #f5f8ff !important;
    color: #1f2937 !important;
    -webkit-text-fill-color: #1f2937 !important;
    caret-color: #e94555 !important;
    opacity: 1 !important;
}
div[data-testid="stTextInput"] input:-webkit-autofill,
div[data-testid="stTextInput"] input:-webkit-autofill:hover,
div[data-testid="stTextInput"] input:-webkit-autofill:focus,
div[data-testid="stTextInput"] input:-webkit-autofill:active {
    -webkit-box-shadow: 0 0 0 1000px #f5f8ff inset !important;
    box-shadow: 0 0 0 1000px #f5f8ff inset !important;
    -webkit-text-fill-color: #1f2937 !important;
    caret-color: #e94555 !important;
}
div[data-testid="stTextInput"] div[data-baseweb="input"],
div[data-testid="stTextInput"] div[data-baseweb="base-input"] {
    background-color: #f5f8ff !important;
}
div[data-testid="stTextInput"] label { color: #263c36 !important; }
</style>
""", unsafe_allow_html=True)

def get_groq_api_key():
    """Read the Groq key from environment, .env, or Streamlit secrets."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except Exception:
        return ""


def generate_ai_reply(history):
    """Generate a reply through Groq's OpenAI-compatible Chat Completions API."""
    key = get_groq_api_key()
    if not key or key in {"PASTE_YOUR_GSK_KEY_HERE", "your_actual_key_here"}:
        return None, (
            "Groq API key is missing. Add GROQ_API_KEY to .streamlit/secrets.toml "
            "or to a local .env file, then restart CareBloom."
        )

    messages = [{
        "role": "system",
        "content": (
            "You are CareBloom, a warm, thoughtful, natural conversational wellness and fitness assistant. "
            "Reply directly in everyday language, remember relevant details from recent turns, and ask useful "
            "follow-up questions when appropriate. Help with general wellness, sleep, hydration, nutrition, "
            "exercise, stress, and everyday topics. Do not diagnose, prescribe medication, or claim to replace "
            "a clinician. For possible emergencies such as chest pain, severe breathing difficulty, stroke "
            "symptoms, or severe bleeding, advise immediate local emergency help. Encourage professional care "
            "when warranted."
        )
    }]

    for item in history[-20:]:
        role = "assistant" if item.get("role") == "assistant" else "user"
        content = str(item.get("content", "")).strip()
        if content:
            messages.append({"role": role, "content": content})

    if len(messages) < 2:
        return None, "Type a message first, then try again."

    model = os.environ.get("GROQ_MODEL", GROQ_MODEL).strip() or GROQ_MODEL
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_completion_tokens": 900
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

    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = json.loads(response.read().decode("utf-8"))

            choices = data.get("choices", [])
            reply = choices[0].get("message", {}).get("content", "") if choices else ""
            if isinstance(reply, str) and reply.strip():
                return reply.strip(), None
            return None, "Groq returned an empty reply. Please try again."

        except urllib.error.HTTPError as error:
            try:
                body = error.read().decode("utf-8", errors="replace")
                detail = json.loads(body).get("error", {}).get("message", "")
            except Exception:
                detail = ""

            if error.code == 401:
                return None, "Groq rejected the API key (HTTP 401). Check GROQ_API_KEY in your secrets file."
            if error.code == 403:
              detail = detail or "No additional error details returned by Groq."
              return None, f"Groq denied this request (HTTP 403). Details: {detail}"
            if error.code == 404:
                return None, (
                    f"Groq model '{model}' was not found or is unavailable (HTTP 404). "
                    "Check the model ID in Groq Console. " + detail
                )
            if error.code == 429:
                return None, (
                    "Groq rate limit or quota reached (HTTP 429). Check your Groq Console usage/limits "
                    "and try again later."
                )
            if error.code in (500, 502, 503, 504) and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            if error.code >= 500:
                return None, f"Groq is temporarily unavailable (HTTP {error.code}). Please try again shortly."
            return None, f"Groq request failed (HTTP {error.code}). {detail}"

        except (http.client.RemoteDisconnected, ConnectionResetError, socket.timeout,
                urllib.error.URLError, TimeoutError, OSError):
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            return None, (
                "Could not connect to Groq after several attempts. Check your internet connection "
                "and try again."
            )
        except (ValueError, KeyError, IndexError, TypeError):
            return None, "Groq returned an unexpected response. Please try again."

    return None, "Groq is temporarily unavailable. Please try again."


def analyze_health_report(report_text="", image_bytes=None, image_mime="image/jpeg"):
    """Explain health-report text or an uploaded image using Groq vision AI."""
    key = get_groq_api_key()
    if not key:
        return None, "GROQ_API_KEY is missing. Check your local .env or Streamlit secrets."

    vision_model = os.environ.get(
        "GROQ_VISION_MODEL",
        "qwen/qwen3.8-27b"
    ).strip()

    instructions = """
You are CareBloom's health-report explainer.
Explain medical report information in plain, calm, easy-to-understand language.

Rules:
- Only report values, units, reference ranges, and findings you can actually read.
- Explain what the report says; do not diagnose a disease or claim certainty.
- Distinguish results explicitly marked high/low from your own interpretation.
- Reference ranges vary by laboratory, age, sex, and other circumstances.
- Never invent missing values, reference ranges, patient details, or findings.
- If the report is blurry, incomplete, or unreadable, say exactly what needs clarification.
- Give general, safe next steps and useful questions to ask a qualified clinician.
- Do not recommend starting, stopping, or changing medicines based on the report.
- Mention urgent professional care when the visible findings or stated symptoms
  suggest a possible emergency; do not delay emergency care for an AI review.
- Protect privacy: do not repeat personal identifiers unnecessarily.

Use these headings:
1. What the report says
2. Results that may need attention
3. What the terms mean
4. Suggested next steps
5. Questions to ask your clinician

If the report contains no readable medical results, explain that clearly.
This is educational information, not a diagnosis.
"""

    user_content = []
    if report_text.strip():
        user_content.append({
            "type": "text",
            "text": "Report text and any additional notes:\n" + report_text.strip()
        })
    else:
        user_content.append({
            "type": "text",
            "text": "Please explain the visible medical report in this image."
        })

    if image_bytes:
        if len(image_bytes) > 10 * 1024 * 1024:
            return None, "This image is over 10 MB. Please upload a smaller image."
        if image_mime not in ("image/jpeg", "image/png", "image/webp"):
            return None, "Please upload a JPG, PNG, or WEBP image."
        encoded_image = base64.b64encode(image_bytes).decode("ascii")
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{image_mime};base64,{encoded_image}"
            }
        })

    payload = {
        "model": vision_model,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.2,
        "max_completion_tokens": 1400
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
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))

        choices = data.get("choices", [])
        answer = (
            choices[0].get("message", {}).get("content", "")
            if choices else ""
        )
        if isinstance(answer, str) and answer.strip():
            return answer.strip(), None
        return None, "The AI returned an empty report explanation. Please try again."

    except urllib.error.HTTPError as error:
        try:
            body = error.read().decode("utf-8", errors="replace")
            detail = json.loads(body).get("error", {}).get("message", body)
        except Exception:
            detail = "No additional details were returned."

        if error.code == 401:
            return None, "Groq rejected the API key. Check your key in local settings."
        if error.code == 403:
            return None, f"Groq denied the report request (HTTP 403): {detail}"
        if error.code == 404:
            return None, (
                f"The vision model '{vision_model}' is unavailable (HTTP 404). "
                "Check Groq Console for a currently supported vision model."
            )
        if error.code == 429:
            return None, "Groq rate limit or quota reached. Please try again later."
        return None, f"Report analysis failed (HTTP {error.code}): {detail}"

    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return None, f"Could not connect to the AI service: {error}"
    except (ValueError, KeyError, IndexError, TypeError):
        return None, "The AI service returned an unexpected response."
def offline_wellness_reply(prompt):
    """Helpful local fallback when the external AI quota/network is unavailable."""
    q = prompt.lower()
    if any(w in q for w in ["weight loss", "lose weight", "lose fat", "slim"]):
        return ("Here are safe starting points: build regular meals around vegetables or fruit, protein, and high-fibre foods; "
                "choose water often; and begin with movement you can sustain, such as a comfortable 10-minute walk. "
                "Avoid crash diets. Your needs depend on age, health, and activity. What does a typical day of eating and movement look like?")
    if any(w in q for w in ["weight gain", "gain weight", "underweight"]):
        return ("For gradual, healthy weight gain, try regular meals and snacks with energy- and nutrient-rich foods such as nuts, "
                "yogurt, eggs, beans, tofu, avocado, and whole grains. Strength exercise can support muscle gain if appropriate. "
                "Unexpected weight loss or difficulty gaining weight deserves a clinician's advice. What foods do you usually eat?")
    if any(w in q for w in ["walk", "steps", "walking"]):
        return ("Start at a comfortable pace and choose a duration you can repeat consistently—even 5–10 minutes is a useful start. "
                "Increase time gradually, wear comfortable shoes, and stop if you feel chest pain, faintness, or unusual breathlessness. "
                "How much do you currently walk in a day?")
    if any(w in q for w in ["diet", "nutrition", "eat", "meal", "protein", "breakfast", "food"]):
        return ("A balanced meal can include vegetables or fruit, a protein source (beans, lentils, eggs, dairy, tofu, fish or meat), "
                "and whole grains or another staple. Affordable options include dal, chickpeas, seasonal produce, eggs, and curd. "
                "Needs vary by person. Do you have a goal, food preference, allergy, or budget I should consider?")
    if any(w in q for w in ["tired", "fatigue", "energy"]):
        return ("Tiredness can have many causes, including sleep, stress, hydration, diet, or a medical issue. Consider regular meals, "
                "water, and a consistent sleep routine, but seek medical advice if it persists or affects daily life. "
                "Get urgent help for severe symptoms such as chest pain or trouble breathing. How long have you felt this way?")
    return ("Thanks for sharing that with CareBloom. I can't reach the AI service right now, so this is a basic offline response rather than an AI-generated answer. "
            "Tell me a little more about your goal, symptoms, or routine and I can offer general wellness information. "
            "For concerning or persistent symptoms, contact a qualified healthcare professional.")

def db():
    c = sqlite3.connect(DB_PATH, timeout=10); c.execute("PRAGMA foreign_keys=ON"); return c

def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL COLLATE NOCASE UNIQUE,password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        c.execute("""CREATE TABLE IF NOT EXISTS progress(id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,entry_date TEXT NOT NULL,water_glasses INTEGER NOT NULL DEFAULT 0,
        steps INTEGER NOT NULL DEFAULT 0,mood TEXT NOT NULL DEFAULT 'Okay',notes TEXT NOT NULL DEFAULT '',
        FOREIGN KEY(user_id) REFERENCES users(id))""")
def hash_password(p):
    salt=os.urandom(16); d=hashlib.pbkdf2_hmac("sha256",p.encode(),salt,240000); return salt.hex()+":"+d.hex()
def verify_password(p,s):
    try:
        sh,dh=s.split(":",1); d=hashlib.pbkdf2_hmac("sha256",p.encode(),bytes.fromhex(sh),240000)
        return hmac.compare_digest(d.hex(),dh)
    except (ValueError,TypeError): return False
def register_user(u,p):
    u=u.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.@-]{3,30}",u): return False,"Username must be 3–30 letters, numbers, dots, underscores, @ or hyphens."
    if len(p)<8: return False,"Password must be at least 8 characters."
    try:
        with db() as c: c.execute("INSERT INTO users(username,password_hash) VALUES(?,?)",(u,hash_password(p)))
        return True,"Account created. You can now sign in."
    except sqlite3.IntegrityError: return False,"That username already exists."
    except sqlite3.Error as e: return False,f"Database error: {e}"
def authenticate_user(u,p):
    with db() as c: row=c.execute("SELECT id,username,password_hash FROM users WHERE username=? COLLATE NOCASE",(u.strip(),)).fetchone()
    return {"id":row[0],"username":row[1]} if row and verify_password(p,row[2]) else None
def require_login():
    if not st.session_state.user:
        st.warning("Please sign in or create an account to use this feature.")
        if st.button("Go to Login / Register"): st.session_state.page="Login / Register"; st.rerun()
        return False
    return True

init_db()
st.markdown("""<style>
:root{--ink:#263b34}.stApp{background:linear-gradient(135deg,#fff9fc,#f4fbf7 55%,#fffdf5);color:var(--ink)}
[data-testid="stSidebar"]{background:#f2faf5;border-right:1px solid #dceee3}
h1,h2,h3,p,label,li,[data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] p{color:#263b34!important}
[data-testid="stSidebar"] div.stButton>button{background:#26313b;color:#fff;border-radius:12px;min-height:2.7rem}
[data-testid="stSidebar"] div.stButton>button p{color:#fff!important}
[data-testid="stSidebar"] div.stButton>button[kind="primary"]{background:#e74750;color:#fff}
[data-testid="stTextInput"] label,[data-testid="stTextInput"] label p{color:#263b34!important}
[data-baseweb="input"]>div,[data-baseweb="textarea"]>div{background:#fff!important;border:1px solid #8a9b92!important}
[data-baseweb="input"] input,[data-baseweb="textarea"] textarea,input,textarea{color:#263b34!important;-webkit-text-fill-color:#263b34!important;caret-color:#263b34!important;background:#fff!important}
input::placeholder,textarea::placeholder{color:#64756c!important;-webkit-text-fill-color:#64756c!important;opacity:1!important}
[data-testid="stChatInput"] textarea{background:#fff!important;color:#263b34!important;-webkit-text-fill-color:#263b34!important;caret-color:#263b34!important}
[data-testid="stChatInput"]>div{background:#fff!important;border:1px solid #e74750!important}
[data-testid="stChatMessage"],[data-testid="stChatMessage"] p,[data-testid="stChatMessage"] li,[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"]{color:#263b34!important}
.cb-hero{padding:1.4rem;border-radius:22px;background:linear-gradient(120deg,#fbe4ef,#e3f5e9);border:1px solid #e8e7df}
</style>""", unsafe_allow_html=True)

if "page" not in st.session_state: st.session_state.page="Home"
if "user" not in st.session_state: st.session_state.user=None
if "chat_history" not in st.session_state: st.session_state.chat_history=[]

with st.sidebar:
    st.markdown("# 🌸 CareBloom"); st.caption("AI Health & Fitness Companion"); st.divider()
    pages=["Home","Login / Register","Health Assistant","Health Report","Wellness Plans","Nearby Hospitals","Progress & Rewards"]
    for p in pages:
        if st.button(p,key="nav_"+p,use_container_width=True,type="primary" if st.session_state.page==p else "secondary"):
            st.session_state.page=p; st.rerun()
    st.divider()
    if st.session_state.user:
        st.success(f"Signed in as **{st.session_state.user['username']}**")
        if st.button("Sign out"): st.session_state.user=None; st.session_state.page="Home"; st.rerun()
    else: st.info("Sign in to save your progress.")

page=st.session_state.page
if page=="Home":
    left,right=st.columns([1.25,1])
    with left:
        st.markdown('<div class="cb-hero">',unsafe_allow_html=True)
        st.title("Small steps. Healthier days. 🌷")
        st.write("Welcome to **CareBloom**, your friendly companion for everyday wellness, movement, mindful habits, and tracking your progress.")
        st.write("A gentle place to build healthy routines—one day at a time.")
        if st.button("Get started",type="primary"): st.session_state.page="Login / Register"; st.rerun()
        st.markdown("</div>",unsafe_allow_html=True)
    with right: st.image("assets/wellness.svg",use_container_width=True)
    st.subheader("Your wellness space")
    cols=st.columns(3)
    for col,emoji,title,desc in zip(cols,["💬","🥗","📈"],["Friendly guidance","Healthy habits","Progress & rewards"],["Ask general wellness questions.","Explore balanced food and movement ideas.","Track daily habits and earn HealthPoints."]):
        with col: st.markdown(f"### {emoji} {title}"); st.write(desc)
    st.info("CareBloom is a student wellness prototype. It does not diagnose, treat, or replace a healthcare professional.")
elif page=="Login / Register":
    st.title("Welcome to CareBloom 🌸"); st.write("Create an account or sign in.")
    t1,t2=st.tabs(["Sign in","Create account"])
    with t1:
        with st.form("signin"):
            u=st.text_input("Username"); p=st.text_input("Password",type="password")
            submit=st.form_submit_button("Sign in",type="primary",use_container_width=True)
        if submit:
            account=authenticate_user(u,p) if u.strip() and p else None
            if account: st.session_state.user=account; st.session_state.page="Home"; st.success("Welcome back!"); st.rerun()
            else: st.error("Sign-in failed. Check your details or create an account.")
    with t2:
        with st.form("register"):
            u=st.text_input("Choose a username"); p=st.text_input("Create password (8+ characters)",type="password"); p2=st.text_input("Confirm password",type="password")
            submit=st.form_submit_button("Create account",type="primary",use_container_width=True)
        if submit:
            if p!=p2: st.error("Passwords do not match.")
            elif not u.strip() or not p: st.error("Fill in all fields.")
            else:
                ok,msg=register_user(u,p)
                (st.success if ok else st.error)(msg)
elif page=="Health Assistant":
    st.title("Your friendly Health Assistant 💬")
    st.write("Have a natural conversation about wellness, fitness, everyday questions, or whatever is on your mind.")
    st.warning("CareBloom offers general information, not medical diagnosis or emergency care.")
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]): st.markdown(item["content"])
    prompt=st.chat_input("Message CareBloom...")
    if prompt:
        st.session_state.chat_history.append({"role":"user","content":prompt})
        with st.spinner("CareBloom is thinking..."):
            reply,error=generate_ai_reply(st.session_state.chat_history)
        if error:
            reply = offline_wellness_reply(prompt)
            if "429" in error:
                st.info("Groq quota/rate limit is unavailable right now. Showing a basic offline wellness suggestion instead.")
            else:
                st.info("Groq AI is unavailable right now. Showing a basic offline wellness suggestion instead.")
            reply += "\n\n*Connection detail: " + error.replace("\n", " ") + "*"
        st.session_state.chat_history.append({"role":"assistant","content":reply or "Please try again."}); st.rerun()

elif page=="Health Report":
    st.title("Health Report Helper 📄")
    st.write(
        "Upload a health report or paste its text to receive "
        "an AI-generated explanation and suggestions."
    )

    if require_login():
        report_text = st.text_area(
            "Paste your health report or describe your health concern",
            height=180,
            key="health_report_text"
        )

        uploaded_file = st.file_uploader(
            "Upload your report (TXT, PDF, PNG, JPG, WEBP)",
            type=["txt", "pdf", "png", "jpg", "jpeg", "webp"],
            key="health_report_upload"
        )

        image_bytes = None
        image_mime = "image/jpeg"
        text_to_analyze = report_text.strip()

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()

            if len(file_bytes) > 10 * 1024 * 1024:
                st.error("Please upload a file smaller than 10 MB.")
                uploaded_file = None

            elif uploaded_file.type == "text/plain":
                extracted_text = file_bytes.decode(
                    "utf-8", errors="replace"
                )
                text_to_analyze = extracted_text.strip()
                st.text_area(
                    "Uploaded report text",
                    value=extracted_text,
                    height=180,
                    disabled=True,
                    key="uploaded_report_preview"
                )

            elif uploaded_file.type == "application/pdf":
                try:
                    from pypdf import PdfReader
                    import io

                    reader = PdfReader(io.BytesIO(file_bytes))
                    extracted_text = "\n".join(
                        page.extract_text() or ""
                        for page in reader.pages
                    ).strip()

                    if extracted_text:
                        text_to_analyze = extracted_text
                        st.text_area(
                            "Extracted PDF text",
                            value=extracted_text,
                            height=220,
                            disabled=True,
                            key="extracted_report_preview"
                        )
                    else:
                        st.warning(
                            "This PDF has no selectable text. "
                            "Upload a page as a PNG/JPG image or paste "
                            "the report text above."
                        )
                except Exception as exc:
                    st.warning(
                        "Could not read this PDF. Please paste its "
                        "text or upload a page as an image."
                    )

            elif uploaded_file.type.startswith("image/"):
                image_bytes = file_bytes
                image_mime = uploaded_file.type

                st.image(
                    image_bytes,
                    caption="Uploaded health report",
                    use_container_width=True
                )

        st.warning(
            "AI explanations are not a medical diagnosis. "
            "Hide personal details such as your name, address, "
            "phone number, and patient ID before uploading."
        )

        if st.button("Analyze my health report", type="primary"):
            if not text_to_analyze and image_bytes is None:
                st.warning(
                    "Please paste report text or upload a readable "
                    "report before continuing."
                )
            else:
                with st.spinner("Analyzing your report..."):
                    try:
                        analysis = analyze_health_report(
                            report_text=text_to_analyze,
                            image_bytes=image_bytes,
                            image_mime=image_mime
                        )

                        if analysis:
                            st.subheader("Report explanation")
                            st.markdown(analysis)
                            st.info(
                                "Discuss the findings with a qualified "
                                "healthcare professional. Do not start, "
                                "stop, or change medication based only "
                                "on this AI explanation."
                            )
                        else:
                            st.error(
                                "The AI returned an empty response. "
                                "Please try again."
                            )

                    except Exception as exc:
                        st.error(
                            "Report analysis failed. Please check your "
                            "API configuration and try again."
                        )
                        st.caption(str(exc))

elif page=="Wellness Plans":
    st.title("Fitness, Wellness & Nutrition 🌿")
    st.write("Ask your fitness or food questions in your own words. CareBloom responds conversationally when the AI service is available.")
    fitness_tab, nutrition_tab = st.tabs(["🏃 Fitness & movement", "🥗 Nutrition & diet"])
    for tab, key, heading, placeholder in [
        (fitness_tab, "fitness_chat", "Ask about fitness", "e.g. I want to build a walking routine..."),
        (nutrition_tab, "nutrition_chat", "Ask about nutrition", "e.g. Suggest affordable protein-rich meals...")
    ]:
        if key not in st.session_state: st.session_state[key] = []
        with tab:
            st.subheader(heading)
            st.caption("You can ask follow-up questions. General information only—not a diagnosis or individualized medical treatment.")
            for item in st.session_state[key][-12:]:
                with st.chat_message(item["role"]): st.markdown(item["content"])
            with st.form(key + "_form", clear_on_submit=True):
                question = st.text_input("Type your question", placeholder=placeholder, key=key+"_input")
                send = st.form_submit_button("Ask CareBloom", type="primary")
            if send:
                if not question.strip():
                    st.warning("Please type your question first.")
                else:
                    st.session_state[key].append({"role":"user","content":question.strip()})
                    with st.spinner("CareBloom is preparing a suggestion..."):
                        reply, error = generate_ai_reply(st.session_state[key])
                    if error:
                        reply = offline_wellness_reply(question)
                        st.info("Groq AI unavailable; showing a basic offline suggestion.")
                        reply += "\n\n*Connection detail: " + error.replace("\n", " ") + "*"
                    st.session_state[key].append({"role":"assistant","content":reply or "Please try again."})
                    st.rerun()
    st.caption("Stop exercising if you experience alarming symptoms. Seek professional advice for persistent concerns or medical conditions.")
elif page=="Nearby Hospitals":
    st.title("Find Nearby Hospitals 🏥"); st.write("Enter a city or area to open a live map search.")
    area=st.text_input("City or area",placeholder="e.g. Bengaluru, India")
    c1,c2=st.columns(2)
    with c1:
        if st.button("Search hospitals on OpenStreetMap",type="primary"):
            if area.strip(): st.link_button("Open hospital map","https://www.openstreetmap.org/search?query="+quote("hospitals near "+area.strip()))
            else: st.warning("Enter a city or area first.")
    with c2:
        if st.button("Search hospitals on Google Maps"):
            if area.strip(): st.link_button("Open Google Maps","https://www.google.com/maps/search/"+quote("hospitals near "+area.strip()))
            else: st.warning("Enter a city or area first.")
elif page=="Progress & Rewards":
    st.title("Progress & HealthPoints 🌱")
    if require_login():
        st.write("Each saved entry earns 10 HealthPoints in this demo.")
        with st.form("progress"):
            d=st.date_input("Date",value=date.today()); water=st.slider("Glasses of water",0,20,6)
            steps=st.number_input("Steps today",min_value=0,max_value=100000,value=3000,step=500)
            mood=st.selectbox("How are you feeling?",["Great","Good","Okay","Low","Stressed"]); notes=st.text_input("Optional note")
            save=st.form_submit_button("Save today's progress",type="primary")
        if save:
            with db() as c: c.execute("INSERT INTO progress(user_id,entry_date,water_glasses,steps,mood,notes) VALUES(?,?,?,?,?,?)",(st.session_state.user["id"],d.isoformat(),water,int(steps),mood,notes.strip()))
            st.success("Progress saved! +10 HealthPoints 🌸")
        with db() as c: entries=c.execute("SELECT entry_date,water_glasses,steps,mood,notes FROM progress WHERE user_id=? ORDER BY entry_date DESC,id DESC LIMIT 30",(st.session_state.user["id"],)).fetchall()
        st.subheader("Recent entries")
        if entries:
            st.dataframe([{"Date":x[0],"Water glasses":x[1],"Steps":x[2],"Mood":x[3],"Notes":x[4]} for x in entries],use_container_width=True,hide_index=True)
            st.metric("HealthPoints earned",len(entries)*10)
        else: st.info("Your saved entries will appear here.")
st.sidebar.caption("Made with care • Student prototype")
