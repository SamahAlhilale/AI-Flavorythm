import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from moviepy.editor import ImageClip, concatenate_videoclips, vfx
import numpy as np
import base64
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
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">AI-Flavorythm</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Flavor-Inspired Art Generator by Alsherazi Club</p>', unsafe_allow_html=True)

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)
    pipe.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    return pipe

model_pipe = load_models()

def generate_video_from_flavor(flavor_description, progress_bar, status_text):
    base_prompt = f"Artistic representation of {flavor_description}, vibrant colors, abstract, food photography style"
    num_images = 4  # Reduced for faster generation
    num_inference_steps = 20
    
    images = []
    for i in range(num_images):
        image = model_pipe(
            base_prompt, 
            num_inference_steps=num_inference_steps, 
            guidance_scale=7.5
        ).images[0]
        images.append(np.array(image))
        progress = (i + 1) / num_images
        progress_bar.progress(progress)
        status_text.text(f"Generating image {i+1}/{num_images}")

    clips = []
    for img in images:
        clip = ImageClip(img).set_duration(1.0)
        clips.append(clip)
    
    final_clip = concatenate_videoclips(clips, method="compose")
    return final_clip.fx(vfx.fadeout, duration=0.3).fx(vfx.fadein, duration=0.3)

def save_and_display_video(video, flavor_description):
    video_filename = f"{re.sub(r'\W+', '_', flavor_description.strip()).lower()}.mp4"
    video.write_videofile(video_filename, codec='libx264', fps=24)
    with open(video_filename, "rb") as f:
        video_bytes = f.read()
    st.video(video_bytes)

# Flavor menu
predefined_flavors = [
    "Citrus & Mint Summer Drink",
    "Dark Chocolate Raspberry Dessert",
    "Caramel Cinnamon Coffee Blend",
    "Lavender Honey Ice Cream"
]

st.sidebar.markdown("### 🍽️ Flavor Palette")
selected_flavor = st.sidebar.radio(
    "Select a flavor inspiration or create your own:",
    ["Create your own"] + predefined_flavors,
    format_func=lambda x: "✨ Create your own flavor" if x == "Create your own" else x
)

if selected_flavor == "Create your own":
    flavor_description = st.text_input("Enter your flavor description:", "")
else:
    flavor_description = st.text_input("Enter your flavor description:", selected_flavor)

if st.button("🎨 Generate Artistic Video"):
    if not flavor_description:
        st.warning("Please enter a flavor description!")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            video = generate_video_from_flavor(flavor_description, progress_bar, status_text)
            save_and_display_video(video, flavor_description)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            status_text.text("Process completed.")
            progress_bar.empty()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 About AI-Flavorythm")
st.sidebar.markdown("AI-Flavorythm is a project by Alsherazi Club that explores the intersection of culinary inspiration and artificial intelligence.")