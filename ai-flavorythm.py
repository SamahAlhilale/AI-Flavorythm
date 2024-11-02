import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from moviepy.editor import ImageClip, concatenate_videoclips, vfx
import traceback
import gc
import numpy as np
import base64
import re

st.set_page_config(layout="wide", page_title="AI-Flavorythm", page_icon="🎨")

# CSS styles remain the same
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

@st.cache_resource(show_spinner=False)
def load_models():
    try:
        with st.spinner('Loading AI model... This might take a minute...'):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float32,  # Using float32 for better compatibility
                safety_checker=None  # Disable safety checker for speed
            )
            pipe = pipe.to(device)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def generate_images(pipe, prompt, num_images=4):
    images = []
    for _ in range(num_images):
        with torch.inference_mode():
            image = pipe(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512
            ).images[0]
            images.append(np.array(image))
    return images

def create_video(images):
    clips = [ImageClip(img).set_duration(1.0) for img in images]
    final_clip = concatenate_videoclips(clips, method="compose")
    return final_clip.fx(vfx.fadeout, duration=0.3).fx(vfx.fadein, duration=0.3)

# Load model at startup
pipe = load_models()

# Flavor menu
predefined_flavors = [
    "Citrus & Mint Summer Drink",
    "Dark Chocolate Raspberry Dessert",
    "Caramel Cinnamon Coffee Blend",
    "Lavender Honey Ice Cream"
]

with st.sidebar:
    st.markdown("### 🍽️ Flavor Palette")
    selected_flavor = st.radio(
        "Select a flavor inspiration or create your own:",
        ["Create your own"] + predefined_flavors,
        format_func=lambda x: "✨ Create your own flavor" if x == "Create your own" else x
    )

# Main area
if selected_flavor == "Create your own":
    flavor_description = st.text_input("Enter your flavor description:", "")
else:
    flavor_description = st.text_input("Enter your flavor description:", selected_flavor)

if st.button("🎨 Generate Artistic Video") and pipe is not None:
    if not flavor_description:
        st.warning("Please enter a flavor description first!")
    else:
        try:
            with st.spinner("Generating your artistic visualization..."):
                prompt = f"Artistic representation of {flavor_description}, vibrant colors, abstract, food photography style"
                images = generate_images(pipe, prompt)
                video = create_video(images)
                
                # Save video
                filename = f"{re.sub(r'\W+', '_', flavor_description.strip()).lower()}.mp4"
                video.write_videofile(filename, codec='libx264', fps=24)
                
                # Display video
                with open(filename, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.code(traceback.format_exc())
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 About AI-Flavorythm")
st.sidebar.markdown("AI-Flavorythm is a project by Alsherazi Club that explores the intersection of culinary inspiration and artificial intelligence.")