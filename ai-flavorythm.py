import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from moviepy.editor import ImageClip, concatenate_videoclips
import numpy as np
import re

st.set_page_config(layout="wide", page_title="AI-Flavorythm", page_icon="🎨")

st.markdown("""
    <style>
    .main { background-color: #1E1E1E; }
    .big-font {
        font-size: 60px !important;
        font-weight: bold;
        color: white;
        text-align: center;
        padding: 20px 0 0 0;
    }
    .subheader {
        font-size: 30px;
        font-style: italic;
        color: white;
        text-align: center;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">AI-Flavorythm</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Flavor-Inspired Art Generator</p>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe = pipe.to(device)
    return pipe

pipe = load_model()

def generate_video(flavor_description, progress_bar):
    prompt = f"Artistic representation of {flavor_description}, vibrant colors, abstract style"
    images = []
    
    for i in range(4):
        image = pipe(prompt, num_inference_steps=20).images[0]
        images.append(np.array(image))
        progress_bar.progress((i + 1) / 4)
    
    clips = [ImageClip(img).set_duration(1.0) for img in images]
    final_clip = concatenate_videoclips(clips)
    filename = f"{re.sub(r'\W+', '_', flavor_description)}.mp4"
    final_clip.write_videofile(filename, fps=24)
    return filename

# Predefined flavors
flavors = [
    "Citrus & Mint Summer Drink",
    "Dark Chocolate Raspberry Dessert",
    "Caramel Cinnamon Coffee Blend",
    "Lavender Honey Ice Cream"
]

# UI
st.sidebar.markdown("### 🍽️ Flavor Palette")
selected = st.sidebar.radio(
    "Choose a flavor or create your own:",
    ["Create your own"] + flavors,
    format_func=lambda x: "✨ Create your own" if x == "Create your own" else x
)

if selected == "Create your own":
    flavor = st.text_input("Enter your flavor description:", "")
else:
    flavor = st.text_input("Enter your flavor description:", selected)

if st.button("🎨 Generate") and flavor:
    progress = st.progress(0)
    try:
        video_file = generate_video(flavor, progress)
        with open(video_file, 'rb') as f:
            st.video(f.read())
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        progress.empty()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 About")
st.sidebar.markdown("AI-Flavorythm transforms flavor descriptions into artistic visualizations.")