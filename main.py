import streamlit as st

st.set_page_config(page_title="SafeRide AI – Project Overview", layout="wide")

st.title("🚦 SafeRide AI — Intelligent Road Safety Monitoring System")
st.markdown("---")

st.markdown("""
## **📌 Why This Project?**

Road accidents and helmet violations are among the major causes of injury and death on Indian roads.  
Most monitoring still relies on **manual surveillance**, which is:

- ❌ Slow  
- ❌ Inaccurate  
- ❌ Expensive  
- ❌ Difficult to scale  

**SafeRide AI** solves this problem using **Computer Vision + LLM Agents + Cloud Automation** to create an intelligent, fully automated road-safety ecosystem.

---

## **🎯 Project Objective**

To build a *real-time AI-powered system* that can:

- Detect **helmet violations**  
- Detect **road accidents**  
- Store detection logs in the cloud  
- Provide **RAG-based chatbot insights**  
- Send **Telegram alerts for accidents**  
- Generate **automatic PDF reports** via AI agents  

The system is designed to assist **police departments, smart city systems, insurance firms, and public safety organizations**.

---

## **✨ Key Features of SafeRide AI**

### 🛵 **1. Helmet Violation Detection**
- Detects bike riders **with/without helmet**
- Works in real-time using YOLO model
- Saves proof images automatically  

### 🚨 **2. Accident Detection**
- Instantly identifies accident scenes  
- Sends **Telegram alerts** with:
  - Location  
  - Timestamp  
  - Confidence  
  - Image evidence  

### 🤖 **3. Agent-Based RAG LLM Chatbot**
Ask questions like:
- *“Show me all accidents from last week”*  
- *“Which camera has most violations?”*  
- *“Email me a report of today’s detections.”*

The agent uses:
- SQL tool  
- Vector search tool  
- S3 bucket fetcher  
- PDF report tool  
- Email sender tool  

### 📝 **4. AI-Generated Reports**
- Summary statistics  
- Charts  
- Proof images  
- System insights  
- Delivered as PDF or email attachment  

### ☁️ **5. Cloud Integration**
- **AWS S3** → stores images  
- **AWS RDS PostgreSQL** → stores logs  
- **EC2** → runs Streamlit application  
- **Telegram Bot API** → alert system  

---

## **💼 Business Use Cases**

### 🚓 **Traffic Law Enforcement**
Automatically identify helmet violations with image proof.

### 🚑 **Emergency Accident Response**
Real-time accident alerts for police & ambulance services.

### 🏙️ **Smart City Applications**
Can integrate with CCTV networks for automated monitoring.

### 🛡️ **Insurance & Investigation**
Provides timestamped, AI-verified accident evidence.

### 📊 **Public Safety Analytics**
Trends and statistics about accidents & helmet usage.

---

## **⚙️ Technical Architecture**

### **1️⃣ YOLO Object Detection**
Classes:
- Helmet  
- No-Helmet  
- Accident  

### **2️⃣ Streamlit Web Application**
- Image/Video upload  
- Live detections  
- Dashboard insights  

### **3️⃣ Cloud Logging**
Each detection stored with:
- Timestamp  
- Location/Camera  
- Class label  
- Confidence  
- Proof image link  

### **4️⃣ RAG LLM Agent**
- Answers queries from logs  
- Uses semantic + SQL retrieval  
- Generates structured responses  

### **5️⃣ Reporting System**
- Creates downloadable PDF with:
  - Data tables  
  - Charts  
  - RAG summary  
  - Proof images  

### **6️⃣ Notification Engine**
- Sends instant accident alerts  
- Telegram bot integration  

---

## **🚀 Benefits of SafeRide AI**

- **Automates traffic monitoring** → less manual work  
- **Faster emergency response** → lives saved  
- **Evidence-backed detection** → reduces disputes  
- **Scalable system** → deployable across city-wide cameras  
- **Data-driven decision making** → better planning & enforcement  
- **24/7 autonomous monitoring**  

---

## **📦 Project Modules**

### **1. YOLO Model Training**
- Dataset collection  
- Annotation  
- Fine-tuning  
- Export to `best.pt`  

### **2. Streamlit App**
- Detection UI  
- Dashboard insights  

### **3. Accident Alert System**
- Sends Telegram notifications  
- Stores panic frames  

### **4. RAG Chatbot**
- Answer queries using DB & S3  
- Agent-based architecture  

### **5. PDF & Email Reporting**
- Weekly/daily automated reports  

---

## **📁 Dataset Summary**
- Helmet / No helmet images  
- Accident frames  
- YOLO txt annotations  
- JPEG/PNG/MP4 formats  

---

## **📑 Deliverables**
- YOLO training notebooks  
- Streamlit application (`main.py`)  
- EC2 deployment script  
- RDS schema + vector search  
- S3 bucket structure  
- Telegram bot setup  
- RAG Chatbot  
- Reporting system  

---

### 💬 *Explore the other pages to try detections, view insights, and ask questions using the RAG Chatbot!*
""")
