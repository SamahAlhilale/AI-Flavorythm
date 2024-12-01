import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from moviepy.editor import ImageClip, concatenate_videoclips, vfx
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
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #3498db;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">AI-Flavorythm</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Flavor-Inspired Art Generator by Alsherazi Club</p>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
    return pipe

pipe = load_model()

def generate_video(flavor, progress_bar, status_text):
    prompt = f"Artistic representation of {flavor}, vibrant colors, abstract style"
    images = []
    
    for i in range(2):
        with torch.inference_mode():
            image = pipe(
                prompt,
                num_inference_steps=8,
                guidance_scale=7.0,
            ).images[0]
            images.append(np.array(image))
            progress = (i + 1) / 2
            progress_bar.progress(progress)
            status_text.text(f"Generating video... {int(progress * 100)}%")
    
    clips = [ImageClip(img).set_duration(0.8) for img in images]
    final_clip = concatenate_videoclips(clips)
    filename = f"{re.sub(r'\W+', '_', flavor)}.mp4"
    final_clip.write_videofile(filename, fps=24)
    return filename

# Flavor menu
flavors = [
    "Citrus & Mint Summer Drink",
    "Dark Chocolate Raspberry Dessert",
    "Caramel Cinnamon Coffee Blend",
    "Lavender Honey Ice Cream"
]

st.sidebar.markdown("### 🍽️ Flavor Palette")
selected = st.sidebar.radio(
    "Choose a flavor:",
    ["Create your own"] + flavors,
    format_func=lambda x: "✨ Create your own" if x == "Create your own" else x
)

if selected == "Create your own":
    flavor = st.text_input("Enter your flavor description:", "")
else:
    flavor = st.text_input("Enter your flavor description:", selected)

if st.button("🎨 Generate Artistic Video"):
    if not flavor:
        st.warning("Please enter a flavor description!")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            video_file = generate_video(flavor, progress_bar, status_text)
            with open(video_file, 'rb') as f:
                st.video(f.read())
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 About AI-Flavorythm")
st.sidebar.markdown("AI-Flavorythm transforms flavors into artistic visualizations.")