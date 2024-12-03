import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from moviepy.editor import ImageClip, concatenate_videoclips, vfx
import numpy as np
import re
import gc

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
    img {
        max-width: 768px !important;
        margin: auto !important;
        display: block !important;
    }
    .progress-bar-text {
        color: white;
        text-align: center;
        font-size: 18px;
        margin-bottom: 10px;
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
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loading_message = st.info('Loading model... This may take a few minutes the first time...')
        
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        if device == "cuda":
            pipe = pipe.to("cuda")
            pipe.enable_attention_slicing()
            
        loading_message.empty()
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

pipe = load_model()
progress_text = st.empty()
progress_bar = st.empty()

def update_progress_callback(step: int, timestep: int, latents: torch.FloatTensor, current_image: int):
    total_progress = (current_image * 8 + (step + 1)) / 16 * 100  
    progress_bar.progress(min(total_progress / 100, 1.0))
    progress_text.markdown(f"<p class='progress-bar-text'>Generation Progress: {min(total_progress, 100):.0f}%</p>", 
                         unsafe_allow_html=True)

def generate_video(flavor, progress_bar, status_text):
    if pipe is None:
        st.error("Model failed to load. Please try refreshing the page.")
        return None
        
    try:
        prompt = f"Artistic representation of {flavor}, vibrant colors, abstract, food photography style"
        images = []
        
        progress_bar.progress(0)
        progress_text.markdown("<p class='progress-bar-text'>Starting generation...</p>", 
                            unsafe_allow_html=True)
        
        for i in range(2):
            with torch.inference_mode():
                image = pipe(
                    prompt,
                    num_inference_steps=8,
                    guidance_scale=7.0,
                    callback=lambda step, timestep, latents: update_progress_callback(step, timestep, latents, i),
                    callback_steps=1
                ).images[0]
                images.append(np.array(image))
        
        status_text.text("Creating video...")
        clips = []
        for img in images:
            clip = ImageClip(img)
            clip = clip.set_duration(0.8)
            clips.append(clip)
            
        final_clip = concatenate_videoclips(clips)
        filename = f"{re.sub(r'\W+', '_', flavor)}.mp4"
        final_clip.write_videofile(filename, fps=24)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return filename
    except Exception as e:
        st.error(f"Generation error: {str(e)}")
        return None

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

if st.button("🎨 Generate Artistic Video", use_container_width=True):
    if not flavor:
        st.warning("Please enter a flavor description!")
    else:
        if pipe is None:
            st.error("Model is not loaded. Please try refreshing the page.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                video_file = generate_video(flavor, progress_bar, status_text)
                if video_file:
                    progress_bar.empty()
                    status_text.empty()
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