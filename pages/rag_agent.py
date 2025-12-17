# app_rag_chatbot_report_full_updated.py
import streamlit as st
import pandas as pd
import io, os, re, requests
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image as PILImage
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from email.message import EmailMessage
import smtplib
import altair as alt

# ---------------- Config ----------------
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2"
TABLE = "detection_logs"
IMAGE_FOLDER = "web_outputs"
ALLOWED_COLUMNS = {"id","timestamp","file_name","mode","result","alert_sent","image_path","extra_info"}

st.set_page_config(layout="wide", page_title="RAG Chatbot & Reports")

# ---------------- Utilities ----------------
def engine_from_inputs(host, port, db, user, pwd):
    uri = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
    return create_engine(uri, future=True)

def ask_ollama(payload: dict, timeout=120):
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# ✅ Updated with _engine to fix caching issue
@st.cache_data(ttl=600)
def fetch_candidates(_engine, start=None, end=None, limit=2000):
    cols = ",".join(sorted(ALLOWED_COLUMNS))
    if start and end:
        sql = f"SELECT {cols} FROM {TABLE} WHERE timestamp >= :start AND timestamp <= :end ORDER BY timestamp DESC LIMIT :limit"
        params = {"start": start, "end": end, "limit": limit}
    else:
        sql = f"SELECT {cols} FROM {TABLE} ORDER BY timestamp DESC LIMIT :limit"
        params = {"limit": limit}
    with _engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)
    return df

def row_to_text(row):
    parts = [f"{c}: {row[c]}" for c in ["id","timestamp","mode","result","file_name","extra_info"] if c in row and pd.notna(row[c])]
    return " | ".join(parts)

def retrieve_top_k(df, query, k=5):
    if df.empty:
        return pd.DataFrame(columns=df.columns)
    docs = df.apply(row_to_text, axis=1).tolist()
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        X = vectorizer.fit_transform(docs)
        qv = vectorizer.transform([query])
        sims = cosine_similarity(qv, X).flatten()
        top_idx = sims.argsort()[::-1][:k]
        top_scores = sims[top_idx]
        result = df.reset_index(drop=True).iloc[top_idx].copy()
        result["_rag_score"] = top_scores
        return result.sort_values("_rag_score", ascending=False).reset_index(drop=True)
    except Exception:
        qterms = [t.lower() for t in re.findall(r"\w+", query)]
        scores = [sum(d.lower().count(t) for t in qterms) for d in docs]
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        result = df.reset_index(drop=True).iloc[top_idx].copy()
        result["_rag_score"] = [scores[i] for i in top_idx]
        return result

def build_rag_prompt(retrieved: pd.DataFrame, user_question: str):
    context_text = "\n".join([f"ROW_ID={r['id']} | {row_to_text(r)}" for _, r in retrieved.iterrows()]) or "(no rows retrieved)"
    prompt = f"""
You are a helpful assistant which MUST answer questions ONLY using the information in the CONTEXT section below.
CONTEXT: Rows from `{TABLE}` with id, timestamp, mode, result, file_name, extra_info, image_path.

Rules:
1) Answer solely from CONTEXT rows.
2) Reference ROW_ID(s) in your answer, e.g., "[source: ROW_ID=123]".
3) If info not in CONTEXT, reply: "I don't know — the requested information is not available."
4) Provide a short summary (1-4 sentences), then a table-like list of matching ROW_IDs with one-line explanation.

CONTEXT:
{context_text}

USER QUESTION:
{user_question}

RESPONSE:
"""
    return prompt

def ask_rag_with_ollama(prompt_text):
    payload = {"model": MODEL, "messages":[{"role":"user","content":prompt_text}], "stream": False}
    resp = ask_ollama(payload)
    if resp.get("error"): return None, f"Ollama error: {resp['error']}"
    message = resp.get("message", {})
    content = message.get("content") if isinstance(message, dict) else resp.get("message")
    if not content: return None, "No content from Ollama."
    return content, None

def save_altair_chart(chart):
    buf = io.BytesIO()
    chart.save(buf, format="png", scale=2)
    buf.seek(0)
    return buf

# ---------------- PDF & Email ----------------
def create_pdf_with_images(df, rag_answer="", charts=None, images_folder=IMAGE_FOLDER):
    charts = charts or []
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    story.append(Paragraph("Detection Report", styles["Title"]))
    story.append(Paragraph(f"Generated on {datetime.utcnow().isoformat()} UTC", styles["Normal"]))
    story.append(Spacer(1,12))

    # Include RAG answer at the top
    if rag_answer:
        story.append(Paragraph("RAG Answer:", styles["Heading2"]))
        story.append(Paragraph(rag_answer.replace("\n","<br />"), styles["Normal"]))
        story.append(Spacer(1,12))

    # Summary table
    if not df.empty:
        data = [df.columns.tolist()] + df.head(20).astype(str).values.tolist()
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.black),
                                   ("BACKGROUND",(0,0),(-1,0),colors.lightgrey)]))
        story.append(table)
        story.append(Spacer(1,12))

    # Charts
    for chart_buf in charts:
        try:
            chart_buf.seek(0)
            story.append(RLImage(chart_buf,width=400,height=250))
            story.append(Spacer(1,12))
        except: continue

    # Images (validate PIL before adding)
    for p in df.get("image_path", []).head(10).dropna():
        local = os.path.join(images_folder, os.path.basename(p))
        if os.path.exists(local):
            try:
                with open(local,"rb") as f: img_bytes=f.read()
                pil = PILImage.open(io.BytesIO(img_bytes)); pil.verify()
                story.append(RLImage(local,width=400,height=250))
                story.append(Spacer(1,12))
            except Exception as e:
                print(f"Skipping invalid image {local}: {e}")
                continue

    doc.build(story)
    buf.seek(0)
    return buf

def send_email_sync(smtp_host, smtp_port, smtp_user, smtp_pass, to_email, subject, body, pdf_buf=None):
    try:
        msg = EmailMessage()
        msg["From"]=smtp_user; msg["To"]=to_email; msg["Subject"]=subject
        msg.set_content(body)
        if pdf_buf:
            msg.add_attachment(pdf_buf.getvalue(), maintype="application", subtype="pdf", filename="detection_report.pdf")
        with smtplib.SMTP_SSL(smtp_host,int(smtp_port)) as server:
            server.login(smtp_user,smtp_pass)
            server.send_message(msg)
        st.success(f"Email sent to {to_email}")
    except Exception as e:
        st.error(f"Email send failed: {e}")

# ---------------- Streamlit UI ----------------
st.title("RAG Chatbot — Detection Logs & Reports")

# Sidebar: DB & Email config
with st.sidebar:
    st.header("Database Connection")
    db_host = st.text_input("DB host","localhost")
    db_port = st.text_input("DB port","5432")
    db_name = st.text_input("DB name")
    db_user = st.text_input("DB user")
    db_pass = st.text_input("DB password", type="password")
    st.markdown("---")
    st.header("SMTP (Email)")
    smtp_host = st.text_input("SMTP host","smtp.gmail.com")
    smtp_port = st.text_input("SMTP port","465")
    smtp_user = st.text_input("SMTP user (sender email)")
    smtp_pass = st.text_input("SMTP password", type="password")
    st.markdown("---")
    st.header("Retrieval Options")
    limit_candidates = st.number_input("Candidate rows to fetch", value=1000, min_value=50, step=50)
    top_k = st.number_input("Top-k to retrieve", value=5, min_value=1, max_value=25)

# Inputs
col1,col2 = st.columns([3,1])
with col1:
    user_question = st.text_area("Ask a question (I will answer ONLY from table `detection_logs`)", height=160)
with col2:
    time_filter = st.selectbox("Time range", ["No filter","Today","Yesterday","Last 7 days","Custom"])
    if time_filter=="Custom":
        start_date = st.date_input("Start date")
        end_date = st.date_input("End date")

action = st.selectbox("Action", ["Preview & Insights","Generate PDF report","Send report by email"])

if "run_clicked" not in st.session_state: st.session_state.run_clicked=False
if st.button("Run"):
    st.session_state.run_clicked=True

# ---------------- Main Logic ----------------
if st.session_state.run_clicked:
    if not all([db_host, db_port, db_name, db_user, db_pass]):
        st.error("Fill DB credentials"); st.stop()
    if not user_question.strip():
        st.error("Type a question"); st.stop()

    engine = engine_from_inputs(db_host, db_port, db_name, db_user, db_pass)

    # Determine time filter
    now = datetime.now(); start=end=None
    if time_filter=="Today":
        start=datetime.combine(now.date(), datetime.min.time())
        end=datetime.combine(now.date(), datetime.max.time())
    elif time_filter=="Yesterday":
        y=now.date()-timedelta(days=1)
        start=datetime.combine(y, datetime.min.time())
        end=datetime.combine(y, datetime.max.time())
    elif time_filter=="Last 7 days":
        start=datetime.combine((now.date()-timedelta(days=7)), datetime.min.time())
        end=datetime.combine(now.date(), datetime.max.time())
    elif time_filter=="Custom":
        start=datetime.combine(start_date, datetime.min.time())
        end=datetime.combine(end_date, datetime.max.time())

    # Fetch candidates
    with st.spinner("Fetching candidate rows..."):
        candidates = fetch_candidates(engine, start=start, end=end, limit=limit_candidates)
    if candidates.empty:
        st.info("No rows retrieved."); st.stop()

    # Retrieve top-k
    with st.spinner("Retrieving top-k relevant rows..."):
        retrieved = retrieve_top_k(candidates, user_question, k=int(top_k))

    # Ask RAG LLM
    st.subheader("LLM Answer (RAG)")
    prompt_text = build_rag_prompt(retrieved, user_question)
    with st.spinner("Asking the LLM..."):
        rag_answer, err = ask_rag_with_ollama(prompt_text)
    if err: st.error(err)
    else: st.markdown(rag_answer)

    # Show top rows
    st.subheader("Retrieved source rows (top results)")
    show_cols=[c for c in ["id","timestamp","mode","result","file_name","extra_info","image_path","_rag_score"] if c in retrieved.columns]
    st.dataframe(retrieved[show_cols].fillna("").astype(str).head(20))

    # Show images
    st.subheader("Images from retrieved rows (first 10)")
    shown=0
    for _,r in retrieved.head(50).iterrows():
        if shown>=10: break
        p=r.get("image_path")
        if isinstance(p,str) and p.strip():
            local=os.path.join(IMAGE_FOLDER, os.path.basename(p))
            if os.path.exists(local):
                try:
                    PILImage.open(local)
                    st.image(local, caption=f"ROW_ID={r['id']} | {r.get('file_name','')}")
                    shown+=1
                except: continue

    # Generate PDF / Email charts
    charts=[]
    if 'timestamp' in candidates.columns:
        candidates['timestamp']=pd.to_datetime(candidates['timestamp'])
        daily=candidates.set_index('timestamp').resample('D').size().reset_index(name='count')
        c1=alt.Chart(daily).mark_line(point=True).encode(x='timestamp:T',y='count:Q').properties(title="Detections Over Time",width=800,height=250)
        st.altair_chart(c1,use_container_width=True)
        charts.append(save_altair_chart(c1))
    if 'mode' in candidates.columns:
        mc=candidates['mode'].value_counts().reset_index(); mc.columns=['mode','count']
        c2=alt.Chart(mc).mark_bar().encode(x='mode:N',y='count:Q').properties(title="Mode Counts",width=600)
        st.altair_chart(c2,use_container_width=True)
        charts.append(save_altair_chart(c2))

    if action in ["Generate PDF report","Send report by email"]:
        pdf_buf=create_pdf_with_images(retrieved, rag_answer=rag_answer or "", charts=charts)
        if action=="Generate PDF report":
            st.download_button("Download PDF report", pdf_buf, file_name="detection_report.pdf")
        else:
            to_email=st.text_input("Recipient email","recipient@example.com")
            if st.button("Send Email"):
                if not all([smtp_host, smtp_port, smtp_user, smtp_pass, to_email]):
                    st.error("Fill SMTP/email details"); st.stop()
                send_email_sync(smtp_host,smtp_port,smtp_user,smtp_pass,to_email,
                                subject="Detection Report",
                                body="Please find attached the detection report.",
                                pdf_buf=pdf_buf)

    st.success("Done.")
