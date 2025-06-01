import os
import streamlit as st
import requests
import time
from dotenv import load_dotenv
from PIL import Image, ImageEnhance # Import ImageEnhance
import io
import base64
from urllib.parse import quote # Import quote for URL encoding

# Load environment variables
load_dotenv()

# Initialize API keys
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
APIFY_TOKEN = os.getenv("APIFY_TOKEN")

# Customizable prompts
SYSTEM_PROMPT = """ You are a professional photojournalist. Your job is to describe the photo you would take to illustrate a news article, blog post, or social media story. Be realistic and direct — this image will be used to represent the story in a real publication.

First, figure out what the article is mainly about. Then write a short, clear description of the scene you would photograph. Follow the correct approach depending on the type of content:

🟦 A) If the story is about a person or group:
- Use medium or wide shots — no close-ups of faces.
- Describe their clothing, posture, expression, and surroundings.
- Keep the look natural, like a real-world photo.

🟩 B) If the story is about an object or animal:
- Keep the background simple and realistic.
- Avoid cinematic effects or over-stylized scenes.
- Focus on sharpness, lighting, and clarity.

🟨 C) If the story is about a concept or situation (e.g. economy, politics, science):
- Use realistic symbols: flags, buildings, people, charts.
- Do not add emotion or drama.
- Keep the image simple, neutral, and believable.

⚠️ Rules:
- No fantasy, surreal, cartoon, or anime styles.
- No glowing, magical lighting, or epic effects.
- Use muted colors, natural light, and a grounded look.
- Avoid close-ups of faces — use medium/wide framing.

🎯 Final Output:
A single photorealistic image prompt that feels like a real photo, not digital art. Keep it realistic, newsworthy, and simple"""

IMAGE_PROMPT_SUFFIX = (
    ", ultra-photorealistic, indistinguishable from a real-world photograph, "
    "captured with a full-frame DSLR camera using a 50mm lens, soft natural lighting, "
    "realistic shadows and highlights, accurate skin texture with pores and imperfections, "
    "true-to-life facial expressions, realistic body proportions, high-resolution detail, "
    "real-world fabric and material textures, balanced exposure, no over-saturation, "
    "natural color grading, sharp focus on primary subjects, slight depth of field, "
    "authentic environmental lighting (daylight, indoor, overcast, etc.), "
    "no stylization or surreal elements, documentary-style composition, "
    "medium or wide shot framing, subtle background blur (if needed), "
    "realistic reflections, ambient occlusion, volumetric depth, no artificial glow, "
    "subtle image grain, clean and believable photojournalistic tone"
)

def generate_scene_prompt(text: str) -> str:
    """Generate a realistic scene description using OpenRouter's GPT-4."""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "openai/gpt-4-turbo-preview",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Create a photojournalist's scene description for this article: {text}"}
            ],
            "max_tokens": 150
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"Error generating scene description: {str(e)}")
        return None

def crop_pollinations_watermark(image_bytes: bytes) -> bytes:
    """Crops the bottom 60 pixels from an image to remove the Pollinations watermark."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
            # Ensure we don't crop more than the image height
            crop_height = max(0, height - 60) # Increased crop amount
            cropped = img.crop((0, 0, width, crop_height))
            output = io.BytesIO()
            # Save in PNG format to preserve potential transparency and quality
            cropped.save(output, format="PNG")
            return output.getvalue()
    except Exception as e:
        st.error(f"Error cropping image: {str(e)}")
        return image_bytes # Return original bytes if cropping fails

def apply_desaturation_contrast_filter(image_bytes: bytes) -> bytes:
    """Applies a subtle desaturation and contrast boost filter to an image."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            # Ensure image is in RGB mode for color operations
            img = img.convert("RGB")

            # Slight desaturation (0.0 is grayscale, 1.0 is original color)
            img = ImageEnhance.Color(img).enhance(0.90)

            # Slight contrast boost (1.0 is original contrast)
            img = ImageEnhance.Contrast(img).enhance(1.10)

            output = io.BytesIO()
            # Save in PNG format
            img.save(output, format="PNG")
            return output.getvalue()
    except Exception as e:
        st.error(f"Error applying filter: {str(e)}")
        return image_bytes # Return original bytes if filtering fails

def generate_image(prompt: str) -> bytes | None:
    """Generate a photorealistic image using Pollinations.AI."""
    try:
        encoded_prompt = quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

        st.write(f"Sending request to Pollinations.AI: {url}") # Show URL for debugging

        response = requests.get(url)

        if response.status_code == 200:
            st.write("Image generated successfully.")
            return response.content # Return image bytes
        else:
            st.error(f"Image generation failed. Status: {response.status_code}. Response text: {response.text}")
            return None

    except Exception as e:
        st.error(f"Error during image generation: {str(e)}")
        return None

def main():
    st.title("News Image Generator")
    st.write("Enter an article or news text to generate a realistic photojournalist-style image.")

    # Text input
    text_input = st.text_area("Enter your article text:", height=200)

    # Add a sidebar for advanced settings
    with st.sidebar:
        st.header("Advanced Settings")
        st.write("Customize the image generation process")

        # Image dimensions (Pollinations might ignore these, but keep for other models)
        image_width = st.selectbox(
            "Image Width",
            options=[768, 1024],
            index=1
        )

        image_height = st.selectbox(
            "Image Height",
            options=[768, 1024],
            index=1
        )

    if st.button("Generate Image"):
        if not text_input:
            st.warning("Please enter some text first.")
            return

        with st.spinner("Generating scene description..."):
            scene_description = generate_scene_prompt(text_input)

        if scene_description:
            st.write("Scene Description:")
            st.write(scene_description)

            # Combine scene description with the suffix
            final_image_prompt = scene_description + IMAGE_PROMPT_SUFFIX

            st.write("Final Prompt for Image Generation:")
            st.code(final_image_prompt)

            with st.spinner("Generating image..."):
                image_data = generate_image(final_image_prompt)

            if image_data:
                st.write("Cropping watermark...")
                cropped_image_data = crop_pollinations_watermark(image_data)

                st.write("Applying realism filter...")
                final_image_data = apply_desaturation_contrast_filter(cropped_image_data)

                # Display the filtered image bytes
                st.image(final_image_data, caption="Generated Image (Filtered)", use_column_width=True)

                # Download button for filtered image bytes (as png)
                st.markdown(f"""
                    <a href="data:image/png;base64,{base64.b64encode(final_image_data).decode()}" download="generated_image.png">
                        <button style="
                            background-color: #4CAF50;
                            color: white;
                            padding: 10px 20px;
                            text-align: center;
                            text-decoration: none;
                            display: inline-block;
                            font-size: 16px;
                            margin: 4px 2px;
                            cursor: pointer;
                            border-radius: 4px;
                        ">
                            Download Image
                        </button>
                    </a>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 