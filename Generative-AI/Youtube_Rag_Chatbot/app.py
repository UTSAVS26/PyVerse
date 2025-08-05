import streamlit as st
import requests
import re

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="🎥 YouTube RAG Chatbot", page_icon="🤖", layout="centered")

# --- Header ---
st.markdown("<h1 style='text-align: center;'>🎥 YouTube RAG Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem;'>Ask questions based on any YouTube video's transcript using Gemini 💡</p>", unsafe_allow_html=True)
st.divider()

# --- YouTube URL Input ---
st.markdown("#### 🔗 Enter YouTube Video URL")
with st.form("extract_form"):
    url = st.text_input("", placeholder="e.g. https://youtu.be/dQw4w9WgXcQ", label_visibility="collapsed")
    submitted = st.form_submit_button("📄 Extract Transcript")

# Extract video ID from URL
def extract_video_id(url):
    patterns = [
        r"(?:https?://)?(?:www\.)?youtu\.be/([^\s&?/]+)",
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([^\s&]+)",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([^\s&]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

if submitted and url:
    video_id = extract_video_id(url)
    if not video_id:
        st.error("❌ Invalid YouTube URL")
    else:
        with st.spinner("🔍 Extracting transcript and generating embeddings..."):
            try:
                response = requests.post(f"{API_URL}/extract", json={"video_id": video_id})
                if response.status_code == 200:
                    st.success(f"✅ Transcript processed. **{response.json()['chunks']}** chunks created.")
                    st.session_state.video_id = video_id
                else:
                    st.error(f"❌ Failed: {response.json().get('detail', 'Unknown error')}")
            except requests.exceptions.RequestException as e:
                st.error(f"🚫 Connection error: {e}")

# --- Question Input ---
if "video_id" in st.session_state:
    st.divider()
    st.markdown("#### 🧠 Ask a Question About the Video")
    question = st.text_input("Type your question:", placeholder="What is the main message of the video?")
    
    if st.button("💬 Get Answer") and question:
        with st.spinner("🤖 Thinking..."):
            try:
                res = requests.post(f"{API_URL}/ask", json={
                    "video_id": st.session_state.video_id,
                    "question": question
                })
                if res.status_code == 200:
                    answer = res.json()['answer']
                    st.success("✅ Gemini says:")
                    st.markdown(f"""
                        <div style='padding: 15px; background-color: #1e1e1e; border-radius: 10px; color: white;'>
                            🧠 <b>{answer}</b>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ Error: {res.json().get('detail', 'Unknown error')}")
            except requests.exceptions.RequestException as e:
                st.error(f"🚫 Connection error: {e}")
else:
    st.info("⏳ Please extract a transcript before asking questions.")
