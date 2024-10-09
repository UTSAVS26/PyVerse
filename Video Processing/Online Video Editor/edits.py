
import moviepy.editor as mp
import streamlit as st
import tempfile
import moviepy.video.fx.all as vfx
import math
st.set_page_config(
    page_title="Mini Video Editor",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="auto",
)
# ======================== This  section will remove the hamburger and watermark and footer and header from streamlit ===========
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            # header {visibility: hidden;}
            footer:after {
                            content:'\u00A9 Rahul-AkaVector. All rights reserved.'; 
	                        visibility: visible;
	                        display: block;
	                        position: relative;
	                        #background-color: red;
	                        padding: 5px;
	                        top: 2px;
                        }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ======================== This  section will remove the hamburger and watermark and footer and header from streamlit ===========

st.title("Online Mini video editor 🤩🤩🤩")
st.markdown("<p style='text-align: right;'>by&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;VECTOR 💻👨‍💻</p>", unsafe_allow_html=True)
st.sidebar.markdown(
        "<h1 style='text-align: center;'>Upload A Video  ☁️➡️</h1>",
        unsafe_allow_html=True)
st.text("""🎥 Create stunning videos on the go with our Online Mini Video Editor! ✂️ Trim, crop, and enhance your footage effortlessly. 
🎞️ Add captivating transitions, 🔤 text, and 🎉 stickers to make your videos pop. 
🎵 Customize audio tracks and share your creations with ease. 
🌟 Unleash your creativity today and bring your videos to life! 🚀💫 
#VideoEditing #CreativeTools #OnlineEditor""")

clip = st.sidebar.file_uploader(" ", type=["mp4"])


