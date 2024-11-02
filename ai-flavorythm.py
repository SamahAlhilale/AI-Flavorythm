import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from moviepy.editor import ImageClip, concatenate_videoclips, vfx
import io
import traceback
import gc
import numpy as np
import base64
import re
import time

st.set_page_config(layout="wide", page_title="AI-Flavorythm", page_icon="🎨")

# CSS styles remain the same...
st.markdown("""
    <style>
    .main { background-color: #1E1E1E; }
    .big-font {
        font-size: 60px !important;
        font-weight: bold;
        color: white;
        text-align: center;
        padding: 20px 0 0 0;
        margin-bottom: 0;
    }
    .subheader {
        font-size: 30px;
        font-style: italic;
        color: white;
        text-align: center;
        margin-top: 5px;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #ecf0f1;
        color: black !important;
    }
    .stProgress > div > div > div > div {
        background-color: #3498db;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">AI-Flavorythm</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Flavor-Inspired Art Generator by Alsherazi Club</p>', unsafe_allow_html=True)

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/stable-diffusion-v1-4"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)
    pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    return device, pipe

device, model_pipe = load_models()

def generate_video_from_flavor(flavor_description, progress_bar, status_text, pipe=model_pipe):
    base_prompt = f"Artistic representation of {flavor_description}, vibrant colors, abstract, food photography style"
    num_images = 6  # Reduced from 8 to 6 for faster generation
    num_inference_steps = 20  # Increased for better quality
    
    images = []
    try:
        for i in range(num_images):
            image = pipe(
                base_prompt, 
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]
            images.append(np.array(image))
            progress = (i + 1) / num_images
            progress_bar.progress(progress)
            status_text.text(f"Generating video... {progress:.0%}")
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            gc.collect()
            raise RuntimeError("Memory error occurred. Please try again.")
        raise e

    clips = []
    for img in images:
        clip = ImageClip(img).set_duration(0.8)  # Shorter duration per clip
        clips.append(clip)
    
    concat_clip = concatenate_videoclips(clips, method="compose")
    final_clip = concat_clip.fx(vfx.fadeout, duration=0.3).fx(vfx.fadein, duration=0.3)
    
    return final_clip

def clean_filename(filename):
    return re.sub(r'\W+', '_', filename.strip()).lower()

def save_and_display_video(video, flavor_description):
    video_filename = f"{clean_filename(flavor_description)}.mp4"
    video.write_videofile(video_filename, codec='libx264', fps=24)
    with open(video_filename, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode('ascii')
    st.markdown(f'''
        <video controls width="512">
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
    ''', unsafe_allow_html=True)

# Flavor menu
st.sidebar.markdown("### 🍽️ Flavor Palette")
predefined_flavors = [
    "Citrus & Mint Summer Drink",
    "Dark Chocolate Raspberry Dessert",
    "Caramel Cinnamon Coffee Blend",
    "Lavender Honey Ice Cream"
]

flavor_menu = st.sidebar.empty()
selected_flavor = flavor_menu.radio(
    "Select a flavor inspiration or create your own:",
    ["Create your own"] + predefined_flavors,
    index=0,
    format_func=lambda x: x if x != "Create your own" else "✨ Create your own flavor"
)

# Main area
if selected_flavor == "Create your own":
    flavor_description = st.text_input("Enter your flavor description:", "", key="flavor_input")
else:
    flavor_description = st.text_input("Enter your flavor description:", selected_flavor, key="flavor_input")

if st.button("🎨 Generate Artistic Video"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Preparing to generate video...")
        video = generate_video_from_flavor(flavor_description, progress_bar, status_text, model_pipe)
        save_and_display_video(video, flavor_description)
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please try again or contact support if the issue persists.")
        st.code(traceback.format_exc())
    finally:
        status_text.text("Process completed.")
        progress_bar.empty()
        torch.cuda.empty_cache()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 About AI-Flavorythm")
st.sidebar.markdown("AI-Flavorythm is a project by Alsherazi Club that explores the intersection of culinary inspiration and artificial intelligence. Our goal is to create a unique sensory experience by translating flavors into visual art.")