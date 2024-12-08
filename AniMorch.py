# install / update libraries before started
# pip install streamlit diffusers tensorflow
# pip install --upgrade streamlit diffusers tensorflow

from asyncio import run
import streamlit as st
from PIL import Image
from diffusers import StableDiffusionPipeline

# Load the pre-trained model
model_name = "runwayml/stable-diffusion-v1-5" 
pipe = StableDiffusionPipeline.from_pretrained(model_name, framework="tf")
pipe = pipe.to("cpu")

# Streamlit UI
st.title("Anime-to-Realistic Image Converter")
st.write("Upload an anime figure image and get a realistic transformation!")

# Upload image
uploaded_image = st.file_uploader("Choose an anime image...", type=["png", "jpg", "jpeg"])

if uploaded_image:
    # Display the uploaded image
    anime_image = Image.open(uploaded_image).convert("RGB")
    st.image(anime_image, caption="Uploaded Anime Image", use_container_width=True)

    # Process the image
    with st.spinner("Converting to a realistic style..."):
        prompt = "a realistic human portrait of the character in the uploaded image, highly detailed, photographic style"
        
        # Convert the anime image to a realistic person using the model
        result = pipe(prompt=prompt, image=anime_image, guidance_scale=7.0, num_inference_steps=25).images[0]
        
        # Display the result
        st.image(result, caption="Realistic Transformation", use_container_width=True)            
