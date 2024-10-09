
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


if clip:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(clip.read())

    st.sidebar.markdown(
        "<h2 style='text-align: center;'>Video Details 📝</h2>",
        unsafe_allow_html=True)

    clip = mp.VideoFileClip(temp_filename)
    w, h = clip.size
    st.sidebar.write(f"Video size 📏 : {w}x{h}")
    duration = clip.duration
    st.sidebar.write(f"Duration ⏱️: {duration.__round__()} seconds")
    st.sidebar.write(f"Duration ⏱️: {(duration/60):.2f} minutes")
    fps = clip.fps
    st.sidebar.write(f"FPS🎞️: {fps.__round__()}")

    st.sidebar.markdown(
        "<h2 style='text-align: center;'>Crop ✂️ </h2>",
        unsafe_allow_html=True)

    start = st.sidebar.number_input("Start Time (in seconds) 🔺")
    end = st.sidebar.number_input("End Time (in seconds) 🔻")

    if end > start:
        clip = clip.subclip(start, end)

    st.sidebar.markdown(
        "<h2 style='text-align: center;'>Video/Audio Fade 💨</h2>",
        unsafe_allow_html=True)

    fade_in = st.sidebar.number_input("Fade in seconds 🔺")
    fade_out = st.sidebar.number_input("Fade out seconds 🔻")
    duration = end - start

    if fade_out < duration and fade_in < duration:
        clip = clip.fx(vfx.fadein, fade_in)
        clip = clip.fx(vfx.fadeout, fade_out)

    st.sidebar.markdown(
        "<h2 style='text-align: center;'>Resizing 🖼️</h2>",
        unsafe_allow_html=True)

    # Resizing
    resize_width = st.sidebar.number_input("Resize Width 🔺", 1 , w , w)
    resize_height = st.sidebar.number_input("Resize Height 🔻", 1, h ,h)
    clip = clip.resize((resize_width, resize_height))

    st.sidebar.markdown(
        "<h2 style='text-align: center;'>Functions 🔩</h2>",
        unsafe_allow_html=True)
    # Mute checkbox
    mute = st.sidebar.checkbox("Mute 🔇")
    if mute:
        clip = clip.fx(vfx.fadeout, duration)

    # Flipping
    flip_horizontal = st.sidebar.checkbox("Flip Horizontal  ↔️")
    flip_vertical = st.sidebar.checkbox("Flip Vertical  ↕️")
    if flip_horizontal:
        clip = clip.fx(vfx.mirror_x)
    if flip_vertical:
        clip = clip.fx(vfx.mirror_y)

    st.sidebar.markdown(
        "<h2 style='text-align: center;'>Adjustments ⚙️</h2>",
        unsafe_allow_html=True)
    # Colour Change
    color = st.sidebar.slider("Colour Change 🎨", 0.0, 2.0, 1.0)
    clip = clip.fx(mp.vfx.colorx, color)

    volume = st.sidebar.slider("Volume Change 🔊", 0.0, 4.0, 1.0)
    clip = clip.volumex(volume)

    # Save video to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_filename = temp_file.name
        clip.write_videofile(temp_filename)

    # Open the temporary file and read its contents as bytes
    with open(temp_filename, "rb") as file:
        video_bytes = file.read()

    # Display the video in Streamlit
    st.header("Edited Video 🔧")
    st.subheader(" ")
    st.video(video_bytes)
    st.markdown(
        "<h6 style='text-align: center;'>To download your edited video, click on the ⋯ (3 dots) at the bottom right corner ⬇️</h6>",
        unsafe_allow_html=True)