import psycopg2
import streamlit as st


st.set_page_config(
    page_title="SafeRide AI | Road Safety Dashboard",
    layout="wide"
)

# -------------------- HEADER --------------------
st.markdown("""
<h1 style='text-align: center;'>🚦 SafeRide AI</h1>
<h4 style='text-align: center; color: gray;'>
AI-Powered Road Safety Monitoring System
</h4>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------- KPI METRICS --------------------

with st.sidebar:
    st.header("Database Connection")

    DB_HOST = st.text_input("DB host")
    DB_PORT = st.text_input("DB port", "5432")
    DB_NAME = st.text_input("DB name")
    DB_USER = st.text_input("DB user")
    DB_PASS = st.text_input("DB password", type="password")

    connect_db = st.button("🔌 Connect")




def get_counts(host, port, dbname, user, password):
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=dbname,
            user=user,
            password=password
        )
        cur = conn.cursor()

        # Total detections
        cur.execute("SELECT COUNT(*) FROM detection_logs")
        total = cur.fetchone()[0]

        # Helmet violations
        cur.execute(
            "SELECT COUNT(*) FROM detection_logs WHERE result = 'No Helmet'"
        )
        helmet_violations = cur.fetchone()[0]

        # Accident detections
        cur.execute(
            "SELECT COUNT(*) FROM detection_logs WHERE result = 'Accident'"
        )
        accidents = cur.fetchone()[0]

        cur.close()
        conn.close()

        return total, helmet_violations, accidents

    except Exception as e:
        st.warning(f"Database error: {e}")
        return 0, 0, 0



if connect_db:
    total, helmet_violations, accidents = get_counts(
        DB_HOST,
        DB_PORT,
        DB_NAME,
        DB_USER,
        DB_PASS
    )
else:
    total, helmet_violations, accidents = 0, 0, 0


col1, col2, col3 = st.columns(3)

col1.metric("🛵 Helmet Violations", helmet_violations)
col2.metric("🚨 Accidents Detected", accidents)
col3.metric("📸 Total Detections", total)



# -------------------- FEATURE CARDS --------------------
st.subheader("✨ System Capabilities")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("### 🪖 Helmet Detection")
    st.write("Detects riders with or without helmet using YOLO in real-time.")

with c2:
    st.markdown("### 🚨 Accident Detection")
    st.write("Instant accident detection with Telegram alert notifications.")

with c3:
    st.markdown("### 🤖 AI RAG Chatbot")
    st.write("Ask questions on detection logs using AI + SQL + Vector Search.")

with c4:
    st.markdown("### 📄 AI Reports")
    st.write("Auto-generated PDF reports with charts and evidence images.")

st.markdown("---")

# -------------------- ARCHITECTURE --------------------
st.subheader("⚙️ System Architecture")

st.markdown("""
YOLO Detection  
⬇  
Streamlit Application  
⬇  
SQL (Logs) + Images       
⬇  
RAG LLM Agent + Telegram Alerts
""")

st.markdown("---")

# -------------------- CALL TO ACTION --------------------
st.subheader("🚀 Get Started")

c1, c2 = st.columns(2)

with c1:
    if st.button("🛵 Go to Detection Page"):
        st.switch_page("pages/detect.py")

with c2:
    if st.button("🤖 Open RAG Chatbot"):
        st.switch_page("pages/rag_agent.py")
