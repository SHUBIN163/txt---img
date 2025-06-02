import os
import streamlit as st
import requests
import time
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageStat, ImageDraw
import io
import json
import numpy as np

# Load environment variables
load_dotenv()

# Initialize API keys
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
APIFY_TOKEN = os.getenv("APIFY_TOKEN")

# Customizable prompts
SYSTEM_PROMPT = """ You are a professional visual researcher tasked with helping an image search system. Your goal is to generate two search phrases that would return the most relevant, realistic, and high-quality photos for a given text input (such as a news article, blog post, or social media post).

For every input, output exactly two lines, formatted like this:

Image 1 search target: [first search phrase]  
Image 2 search target: [second search phrase]
Follow these rules when writing your search phrases:

Be specific and avoid generic terms like “people” or “technology.”
Focus on the key subjects, actions, or objects mentioned in the text.
Use natural, simple language — like what you would type into Google Images.
Make each phrase different but related to the same topic.
Make sure both phrases would return real, high-quality, news-style photos.
Don’t include any extra words like “photo of” or “image of.” Just describe the scene or object directly.
No stylized or AI-related terms — focus on real-world imagery.
Example:
For a post about Elon Musk revealing a new Tesla model at a press event, your output could be:

Image 1 search target: Elon Musk speaking at Tesla press conference  
Image 2 search target: Tesla Model Y on display at event
Only return the two search lines. No additional commentary or formatting."""

def generate_search_phrases(text: str) -> tuple[str, str] | None:
    """Generate two search phrases using OpenRouter's GPT-4."""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "openai/gpt-4-turbo-preview",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate two search phrases for this text: {text}"}
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
        
        # Parse the response to extract both search phrases
        content = data["choices"][0]["message"]["content"].strip()
        lines = content.split('\n')
        
        phrase1 = lines[0].replace("Image 1 search target:", "").strip()
        phrase2 = lines[1].replace("Image 2 search target:", "").strip()
        
        return phrase1, phrase2
    except Exception as e:
        st.error(f"Error generating search phrases: {str(e)}")
        return None

def has_watermark(img: Image.Image) -> bool:
    """
    Detect if an image has a watermark by analyzing corner regions for high contrast patterns.
    Returns True if a watermark is detected, False otherwise.
    """
    try:
        # Convert to grayscale
        grayscale = img.convert("L")
        width, height = img.size
        
        # Define corner regions to check (100x100 pixels)
        corner_size = min(100, width // 4, height // 4)  # Ensure corner size is reasonable
        corners = [
            grayscale.crop((0, 0, corner_size, corner_size)),  # top-left
            grayscale.crop((width - corner_size, 0, width, corner_size)),  # top-right
            grayscale.crop((0, height - corner_size, corner_size, height)),  # bottom-left
            grayscale.crop((width - corner_size, height - corner_size, width, height))  # bottom-right
        ]
        
        # Create a debug image to show the regions being checked
        debug_img = img.copy()
        draw = ImageDraw.Draw(debug_img)
        
        # Draw rectangles around the regions being checked
        regions = [
            (0, 0, corner_size, corner_size),  # top-left
            (width - corner_size, 0, width, corner_size),  # top-right
            (0, height - corner_size, corner_size, height),  # bottom-left
            (width - corner_size, height - corner_size, width, height)  # bottom-right
        ]
        
        watermark_detected = False
        detection_results = []
        
        # Check each corner for watermark-like patterns
        for i, (region, corner) in enumerate(zip(corners, regions)):
            # Calculate statistics for the region
            stat = ImageStat.Stat(region)
            
            # Check for high contrast (difference between max and min values)
            contrast = stat.extrema[0][1] - stat.extrema[0][0]
            std_dev = stat.stddev[0]
            
            # Store detection results
            result = {
                "region": f"Corner {i+1}",
                "contrast": contrast,
                "std_dev": std_dev,
                "detected": contrast > 100 and std_dev > 50
            }
            detection_results.append(result)
            
            # Draw rectangle with color based on detection
            color = (255, 0, 0) if result["detected"] else (0, 255, 0)  # Red if watermark detected, green if clean
            draw.rectangle(corner, outline=color, width=3)
            
            if result["detected"]:
                watermark_detected = True
        
        # Display the debug image and detection results
        st.write("🔍 Image Analysis:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(debug_img, caption="Image Analysis", use_column_width=True)
        
        with col2:
            st.write("Analysis Results:")
            for result in detection_results:
                status = "⚠️ Possible text/logo" if result["detected"] else "✅ Clean"
                st.write(f"{result['region']}: {status}")
        
        if watermark_detected:
            st.warning("⚠️ Note: This image may contain text, logos, or watermarks in the corners. Please review before use.")
        
        return False  # Always return False to allow the image to be used
        
    except Exception as e:
        st.write(f"Warning: Error during image analysis: {str(e)}")
        return False  # If detection fails, assume no watermark

def search_google_images(search_phrase: str) -> str | None:
    """Search for images using Apify's Google Images Scraper with polling."""
    try:
        st.write(f"Starting Apify Google Images search run for: '{search_phrase}'")

        # 1. Start the actor run asynchronously
        run_start_url = f"https://api.apify.com/v2/acts/hooli~google-images-scraper/runs?token={APIFY_TOKEN}"
        
        payload = {
            "queries": [search_phrase],
            "maxResultsPerQuery": 10,  # Increased to have more options for high-res images
            "includeHtml": False,
            "includeOgp": False
        }

        start_response = requests.post(
            run_start_url,
            headers={
                "Content-Type": "application/json"
            },
            json=payload
        )
        start_response.raise_for_status()
        run_data = start_response.json()
        run_id = run_data["data"]["id"]
        st.write(f"Apify run started with ID: {run_id}")

        # 2. Poll for run completion
        run_status_url = f"https://api.apify.com/v2/actor-runs/{run_id}?token={APIFY_TOKEN}"
        st.write("Polling for run completion...")
        while True:
            status_response = requests.get(run_status_url)
            status_response.raise_for_status()
            status_data = status_response.json()
            
            run_status = status_data["data"]["status"]
            st.write(f"Run status: {run_status}...")

            if run_status in ["SUCCEEDED", "FAILED", "ABORTED", "TIMED_OUT"]:
                break
            
            time.sleep(3) # Wait for 3 seconds before polling again

        if run_status != "SUCCEEDED":
            st.error(f"Apify run failed with status: {run_status}")
            return None

        st.write("Apify run succeeded. Fetching results...")

        # 3. Fetch results from the dataset
        dataset_items_url = f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items?token={APIFY_TOKEN}"
        results_response = requests.get(dataset_items_url)
        results_response.raise_for_status()
        
        if not results_response.text:
            st.error("Received empty dataset items response from Apify.")
            return None

        results_data = results_response.json()
        
        # Filter for high-resolution images
        MIN_WIDTH = 800
        MIN_HEIGHT = 600
        
        high_res_images = []
        for item in results_data:
            # First try to get the full-size image URL
            image_url = item.get("imageUrl")
            if not image_url:
                # Fall back to thumbnail only if full-size URL is not available
                image_url = item.get("thumbnailUrl")
                if image_url:
                    st.write("Warning: Using thumbnail URL as fallback - image quality may be reduced")
            
            if not image_url:
                continue  # Skip if no URL is available
                
            # Get image dimensions from metadata
            width = item.get("imageWidth", 0)
            height = item.get("imageHeight", 0)
            
            if width >= MIN_WIDTH and height >= MIN_HEIGHT:
                try:
                    # Validate the image by downloading it
                    response = requests.get(image_url, timeout=10)
                    response.raise_for_status()
                    
                    # Verify it's actually an image
                    content_type = response.headers.get('content-type', '')
                    if not content_type.startswith('image/'):
                        st.write(f"Skipping non-image content: {content_type}")
                        continue
                        
                    # Open image to verify dimensions
                    img = Image.open(io.BytesIO(response.content))
                    actual_width, actual_height = img.size
                    
                    # Verify actual dimensions match metadata
                    if actual_width >= MIN_WIDTH and actual_height >= MIN_HEIGHT:
                        high_res_images.append({
                            "url": image_url,
                            "width": actual_width,
                            "height": actual_height,
                            "content_type": content_type,
                            "image": img  # Store the image object for later watermark check
                        })
                        st.write(f"Found high-res image: {actual_width}x{actual_height} ({content_type})")
                    else:
                        st.write(f"Skipping image with mismatched dimensions: metadata={width}x{height}, actual={actual_width}x{actual_height}")
                        
                except Exception as e:
                    st.write(f"Error validating image {image_url}: {str(e)}")
                    continue
        
        if not high_res_images:
            st.warning(f"No suitable high-resolution images found (min {MIN_WIDTH}x{MIN_HEIGHT}) for search phrase: {search_phrase}")
            return None
            
        # Sort by resolution (width * height) and take the highest
        best_image = max(high_res_images, key=lambda x: x["width"] * x["height"])
        st.write(f"Selected best image: {best_image['width']}x{best_image['height']} ({best_image['content_type']})")
        
        # Check for watermark only on the best image
        st.write("🔍 Analyzing selected image...")
        has_watermark(best_image["image"])  # Just show the analysis, don't block the image
        
        return best_image["url"]

    except requests.exceptions.RequestException as e:
        st.error(f"Error during Apify Google Images search polling process: {e}")
        st.error(f"Last known response content: {response.text if 'response' in locals() and response.text else 'N/A'}")
        return None
    except json.JSONDecodeError as e:
         st.error(f"Error parsing Apify Google Images response as JSON during polling or results fetch: {e}.")
         last_response_text = 'N/A'
         if 'status_response' in locals() and status_response.text:
             last_response_text = status_response.text
         elif 'results_response' in locals() and results_response.text:
             last_response_text = results_response.text
         st.error(f"Raw response text: {last_response_text}")
         return None
    except KeyError as e:
         st.error(f"Error parsing Apify Google Images response: Missing key {e}. Response structure might have changed.")
         last_data = 'N/A'
         if 'status_data' in locals():
             last_data = status_data
         elif 'results_data' in locals():
             last_data = results_data
         st.error(f"Response data: {last_data}")
         return None
    except Exception as e:
        st.error(f"An unexpected error occurred during Google Images search: {str(e)}")
        return None

def download_image(url: str) -> bytes | None:
    """Download an image from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        st.error(f"Error downloading image: {str(e)}")
        return None

def create_and_save_collage(image1_bytes: bytes, image2_bytes: bytes, output_path: str = "output/collage.jpg", target_height: int = 1024, gutter_width: int = 16) -> str | None:
    """Creates a side-by-side collage from two images, resizes, crops, and saves it."""
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            st.write(f"Created output directory: {output_dir}")

        img1 = Image.open(io.BytesIO(image1_bytes)).convert("RGB")
        img2 = Image.open(io.BytesIO(image2_bytes)).convert("RGB")

        # Resize to target height while maintaining aspect ratio
        img1_resized = img1.resize((int(img1.width * target_height / img1.height), target_height), Image.Resampling.LANCZOS)
        img2_resized = img2.resize((int(img2.width * target_height / img2.height), target_height), Image.Resampling.LANCZOS)

        # Smart Crop (from left/right) to make widths roughly equal or fit total width
        max_width_after_resize = 1024 # Aiming for max width 1024 per image after resize

        def smart_crop(image, target_w, target_h):
            current_w, current_h = image.size
            if current_w > target_w:
                # Crop symmetrically from left/right
                left = (current_w - target_w) // 2
                right = left + target_w
                # Height remains the same as it was already resized
                return image.crop((left, 0, right, current_h))
            return image # No cropping needed if already within target width

        img1_cropped = smart_crop(img1_resized, max_width_after_resize, target_height)
        img2_cropped = smart_crop(img2_resized, max_width_after_resize, target_height)

        # Create collage
        collage_width = img1_cropped.width + img2_cropped.width + gutter_width
        collage_height = target_height # Both images are already resized to target_height

        collage = Image.new("RGB", (collage_width, collage_height), (255, 255, 255)) # White background

        collage.paste(img1_cropped, (0, 0))
        collage.paste(img2_cropped, (img1_cropped.width + gutter_width, 0))

        # Save the collage
        collage.save(output_path, "JPEG")
        st.write(f"Collage saved to {output_path}")
        return output_path

    except Exception as e:
        st.error(f"Error creating or saving collage: {str(e)}")
        return None

def main():
    st.title("Article Image Generator")
    st.write("Enter an article or text to generate two complementary images.")

    # Text input
    text_input = st.text_area("Enter your article text:", height=200)

    if st.button("Generate Images"):
        if not text_input:
            st.warning("Please enter some text first.")
            return

        # Create a container for the entire process
        with st.container():
            # Step 1: Generate search phrases
            st.subheader("Step 1: Generating Search Phrases")
            with st.spinner("Analyzing your text..."):
                search_phrases = generate_search_phrases(text_input)

            if search_phrases:
                phrase1, phrase2 = search_phrases
                st.success("Search phrases generated successfully!")
                st.info("Search Phrases:")
                st.write(f"1. {phrase1}")
                st.write(f"2. {phrase2}")

                # Step 2: Search for images
                st.subheader("Step 2: Finding High-Quality Images")
                
                # Create two columns for image search progress
                col1, col2 = st.columns(2)
                
                image_url1 = None
                image_url2 = None
                image_data1 = None
                image_data2 = None

                # Search for first image
                with col1:
                    st.write("🔍 First Image Search")
                    with st.spinner("Searching..."):
                        image_url1 = search_google_images(phrase1)

                # Search for second image
                with col2:
                    st.write("🔍 Second Image Search")
                    with st.spinner("Searching..."):
                        image_url2 = search_google_images(phrase2)

                # Step 3: Download images
                st.subheader("Step 3: Processing Images")
                progress_bar = st.progress(0)
                
                if image_url1:
                    st.write("📥 Downloading first image...")
                    image_data1 = download_image(image_url1)
                    progress_bar.progress(50)

                if image_url2:
                    st.write("📥 Downloading second image...")
                    image_data2 = download_image(image_url2)
                    progress_bar.progress(100)

                # Step 4: Create collage
                if image_data1 and image_data2:
                    st.subheader("Step 4: Creating Collage")
                    with st.spinner("Combining images..."):
                        collage_path = create_and_save_collage(image_data1, image_data2)
                        if collage_path:
                            st.success("Collage created successfully!")
                            st.image(collage_path, caption="Final Collage", use_column_width=True)
                            
                            # Add download button for the collage
                            with open(collage_path, "rb") as file:
                                btn = st.download_button(
                                    label="Download Collage",
                                    data=file,
                                    file_name="collage.jpg",
                                    mime="image/jpeg"
                                )
                else:
                    st.error("Failed to create collage. Please try again with different search phrases.")

if __name__ == "__main__":
    main() 