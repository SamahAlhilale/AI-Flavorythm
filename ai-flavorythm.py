import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from moviepy.editor import ImageClip, concatenate_videoclips, vfx
import numpy as np
import re

st.set_page_config(layout="wide", page_title="AI-Flavorythm", page_icon="🎨")

# CSS styles
st.markdown("""
    <style>
    .main {
        background-color: #1E1E1E;
    }
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
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
    return pipe

pipe = load_model()

def generate_video_from_flavor(flavor_description, progress_bar, status_text):
    base_prompt = f"Artistic representation of {flavor_description}, vibrant colors, abstract, food photography style"
    num_images = 3  
    num_inference_steps = 12  
    
    images = []
    try:
        for i in range(num_images):
            status_text.text(f"✨ Creating artistic interpretation {i+1} of {num_images} ({(i+1)/num_images*100:.0f}%)")
            with torch.inference_mode():
                image = pipe(
                    base_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=7.0,
                ).images[0]
                images.append(np.array(image))
                progress = (i + 1) / num_images
                progress_bar.progress(progress)

        status_text.text("🎬 Composing video sequence...")
        clips = []
        for img in images:
            clip = ImageClip(img).set_duration(0.8)  
            clips.append(clip)
        
        concat_clip = concatenate_videoclips(clips, method="compose")
        final_clip = concat_clip.fx(vfx.fadeout, duration=0.3).fx(vfx.fadein, duration=0.3)
        
        status_text.text("📼 Finalizing your artistic visualization...")
        return final_clip
        
    except Exception as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e

def save_and_display_video(video, flavor_description, status_text):
    video_filename = f"{re.sub(r'\W+', '_', flavor_description)}.mp4"
    video.write_videofile(video_filename, codec='libx264', fps=24)
    with open(video_filename, 'rb') as f:
        st.video(f.read())

# Flavor menu
st.sidebar.markdown("### 🍽️ Flavor Palette")
predefined_flavors = [
    "Citrus & Mint Summer Drink",
    "Dark Chocolate Raspberry Dessert",
    "Caramel Cinnamon Coffee Blend",
    "Lavender Honey Ice Cream"
]

selected_flavor = st.sidebar.radio(
    "Select a flavor inspiration or create your own:",
    ["Create your own"] + predefined_flavors,
    format_func=lambda x: "✨ Create your own flavor" if x == "Create your own" else x
)

if selected_flavor == "Create your own":
    flavor_description = st.text_input("Enter your flavor description:", "", key="flavor_input")
else:
    flavor_description = st.text_input("Enter your flavor description:", selected_flavor, key="flavor_input")

if st.button("🎨 Generate Artistic Video"):
    if not flavor_description:
        st.warning("Please enter a flavor description first!")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🎨 Initializing artistic generation...")
            video = generate_video_from_flavor(flavor_description, progress_bar, status_text)
            save_and_display_video(video, flavor_description, status_text)
            status_text.text("✨ Generation complete! Enjoy your flavor visualization!")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 About AI-Flavorythm")
st.sidebar.markdown("AI-Flavorythm is a project by Alsherazi Club that explores the intersection of culinary inspiration and artificial intelligence. Our goal is to create a unique sensory experience by translating flavors into visual art.")