import os
import streamlit as st
import requests
import time
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageStat, ImageDraw
import io
import json
import numpy as np
from typing import List
from image_selector import ImageSelector

# Load environment variables
load_dotenv()

# Initialize API keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
APIFY_TOKEN = os.getenv("APIFY_TOKEN")

# Initialize ImageSelector
image_selector = ImageSelector()

# Customizable prompts
SYSTEM_PROMPT = """ You are a professional visual researcher tasked with helping an image search system. Your goal is to generate two search phrases that would return the most relevant, realistic, and high-quality photos for a given text input (such as a news article, blog post, or social media post).

For every input, output exactly two lines, formatted like this:

Image 1 search target: [first search phrase]  
Image 2 search target: [second search phrase]
Follow these rules when writing your search phrases:

Be specific and avoid generic terms like \"people\" or \"technology.\"\nFocus on the key subjects, actions, or objects mentioned in the text.\nUse natural, simple language — like what you would type into Google Images.\nMake each phrase different but related to the same topic.\nMake sure both phrases would return real, high-quality, news-style photos.\nDon't include any extra words like \"photo of\" or \"image of.\" Just describe the scene or object directly.\nNo stylized or AI-related terms — focus on real-world imagery.\nExample:\nFor a post about Elon Musk revealing a new Tesla model at a press event, your output could be:\n\nImage 1 search target: Elon Musk speaking at Tesla press conference  \nImage 2 search target: Tesla Model Y on display at event\nOnly return the two search lines. No additional commentary or formatting."""

def generate_search_phrases(text: str) -> tuple[str, str] | None:
    """Generate two search phrases using OpenRouter's GPT-4."""
    try:
        if not OPENROUTER_API_KEY:
            st.error("OPENROUTER_API_KEY not found in environment variables. Please set it up.")
            return None
            
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
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

def search_google_images(search_phrase: str) -> List[str]:
    """Search for images using Apify's Google Images Scraper and return a list of up to 5 high-res image URLs."""
    image_urls = []
    try:
        st.write(f"Starting Apify Google Images search run for: '{search_phrase}'")

        # 1. Start the actor run asynchronously
        run_start_url = f"https://api.apify.com/v2/acts/hooli~google-images-scraper/runs?token={APIFY_TOKEN}"
        
        payload = {
            "queries": [search_phrase],
            "maxResultsPerQuery": 10,  # Fetch an initial batch, will process all available dataset items later
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
        
        status_placeholder = st.empty() # Create a placeholder for the status message
        status_placeholder.write("Polling for run completion...")

        while True:
            status_response = requests.get(run_status_url, timeout=90) # Polling timeout
            status_response.raise_for_status()
            status_data = status_response.json()
            
            run_status = status_data["data"]["status"]
            status_placeholder.write(f"Run status: {run_status}...") # Update the placeholder without spinner

            if run_status in ["SUCCEEDED", "FAILED", "ABORTED", "TIMED_OUT"]:
                break
            
            time.sleep(1.5) # Wait before polling again

        if run_status != "SUCCEEDED":
            status_placeholder.error(f"Apify run failed with status: {run_status}") # Show final status
            return image_urls

        status_placeholder.success("Apify run succeeded.") # Show final status

        st.write("Fetching results...")

        # 3. Fetch ALL results from the dataset and find up to 5 valid images
        dataset_items_url = f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items?token={APIFY_TOKEN}"
        # Fetch ALL available items
        results_response = requests.get(dataset_items_url, params={'limit': 99999}) # Fetch a high limit to get all items
        results_response.raise_for_status()
        
        if not results_response.text:
            st.error("Received empty dataset items response from Apify.")
            return image_urls

        results_data = results_response.json()
        
        # Filter for high-resolution images and collect up to 5 VALID image URLs
        MIN_WIDTH = 800
        MIN_HEIGHT = 600
        MAX_VALID_CANDIDATES = 5 
        
        st.write(f"Processing {len(results_data)} potential image candidates from dataset...")

        for item in results_data:
            if len(image_urls) >= MAX_VALID_CANDIDATES:
                break # Stop once we have enough valid images
                
            # First try to get the full-size image URL
            image_url = item.get("imageUrl")
            if not image_url:
                # Fall back to thumbnail only if full-size URL is not available
                image_url = item.get("thumbnailUrl")
                if image_url:
                    st.write("Warning: Using thumbnail URL as fallback - image quality may be reduced")
            
            if not image_url:
                continue  # Skip if no URL is available
                
            # Get image dimensions from metadata (can be inaccurate)
            metadata_width = item.get("imageWidth", 0)
            metadata_height = item.get("imageHeight", 0)
            
            # Prioritize images with metadata indicating sufficient size, but always validate with PIL
            if metadata_width >= MIN_WIDTH and metadata_height >= MIN_HEIGHT:
                try:
                    # Download the full image to validate with PIL
                    response = requests.get(image_url, timeout=10)
                    response.raise_for_status()

                    # Verify it's actually an image content type
                    content_type = response.headers.get('content-type', '')
                    if not content_type.startswith('image/'):
                         st.write(f"Skipping non-image content: {content_type} for {image_url}")
                         continue

                    try:
                         img = Image.open(io.BytesIO(response.content))
                         actual_width, actual_height = img.size
                    except Exception as img_e:
                         st.write(f"Error opening image with PIL {image_url}: {img_e}")
                         continue # Skip this image if PIL fails to open it

                    # Verify actual dimensions match the minimum requirement
                    if actual_width >= MIN_WIDTH and actual_height >= MIN_HEIGHT:
                        image_urls.append(image_url)
                        st.write(f"Found valid high-res image candidate: {actual_width}x{actual_height} ({content_type}) from {image_url}")
                    else:
                        st.write(f"Skipping image with insufficient actual dimensions: {actual_width}x{actual_height} from {image_url}")
                        
                except Exception as e:
                    # Catching potential errors during image validation/download for a single candidate
                    st.write(f"Error validating or downloading image candidate {image_url}: {str(e)}")
                    # DO NOT return here, continue to the next image
        
        if not image_urls:
            st.warning(f"No suitable high-resolution image candidates found (min {MIN_WIDTH}x{MIN_HEIGHT}) for search phrase: {search_phrase}")
        else:
            st.success(f"Successfully collected {len(image_urls)} valid image candidates.")
            
        return image_urls # Return the list of found valid image URLs

    except requests.exceptions.RequestException as e:
        st.error(f"Error during Apify Google Images search polling or results fetch: {e}")
        # Check if response object exists before accessing .text
        last_response_text = 'N/A'
        if 'start_response' in locals() and start_response and start_response.text:
             last_response_text = start_response.text
        elif 'status_response' in locals() and status_response and status_response.text:
             last_response_text = status_response.text
        elif 'results_response' in locals() and results_response and results_response.text:
             last_response_text = results_response.text
        st.error(f"Last known response content: {last_response_text}")
        return image_urls # Return partial results or empty list on error
    except json.JSONDecodeError as e:
         st.error(f"Error parsing Apify Google Images response as JSON during polling or results fetch: {e}.")
         last_response_text = 'N/A'
         if 'status_response' in locals() and status_response and status_response.text:
             last_response_text = status_response.text
         elif 'results_response' in locals() and results_response and results_response.text:
             last_response_text = results_response.text
         st.error(f"Raw response text: {last_response_text}")
         return image_urls # Return partial results or empty list on error
    except KeyError as e:
         st.error(f"Error parsing Apify Google Images response: Missing key {e}. Response structure might have changed.")
         last_data = 'N/A'
         if 'status_data' in locals():
             last_data = status_data
         elif 'results_data' in locals():
             last_data = results_data
         st.error(f"Response data: {last_data}")
         return image_urls # Return partial results or empty list on error
    except Exception as e:
        st.error(f"An unexpected error occurred during Google Images search: {str(e)}")
        return image_urls # Return partial results or empty list on error

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
    st.write("Enter an article or text to generate two complementary images.") # Reverted description

    # Text input
    text_input = st.text_area("Enter your article text:", height=200)

    if st.button("Generate Images"): # Reverted button text
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

                # Step 2: Find Candidate Images for Phrase 1
                st.subheader(f"Step 2: Finding Candidate Images for '{phrase1}'")
                candidate_image_urls_1 = []
                
                with st.spinner(f"Searching for up to 10 image candidates using phrase: '{phrase1}'..."):
                    candidate_image_urls_1 = search_google_images(phrase1)

                if not candidate_image_urls_1:
                    st.error(f"No suitable image candidates found for '{phrase1}'. Cannot proceed with image generation.")
                    return # Stop if no candidates are found for the first phrase

                st.success(f"Found {len(candidate_image_urls_1)} image candidates for '{phrase1}'.")

                # Step 3: Find Candidate Images for Phrase 2
                st.subheader(f"Step 3: Finding Candidate Images for '{phrase2}'")
                candidate_image_urls_2 = []
                
                with st.spinner(f"Searching for up to 10 image candidates using phrase: '{phrase2}'..."):
                    candidate_image_urls_2 = search_google_images(phrase2)
                    
                if not candidate_image_urls_2:
                     st.error(f"No suitable image candidates found for '{phrase2}'. Cannot proceed with image generation.")
                     return # Stop if no candidates are found for the second phrase

                st.success(f"Found {len(candidate_image_urls_2)} image candidates for '{phrase2}'.")
                
                # Step 4: Select the Best Image for Phrase 1 using AI
                st.subheader(f"Step 4: Selecting the Best Image for '{phrase1}' with AI")
                
                # Ensure we have at least 5 candidates for GPT-4 input (pad if necessary)
                while len(candidate_image_urls_1) < 5 and candidate_image_urls_1:
                     candidate_image_urls_1.append(candidate_image_urls_1[0]) # Pad with duplicates
                while len(candidate_image_urls_1) < 5:
                     # This case should ideally not be reached if search_google_images returns at least one image on success
                     st.error(f"Insufficient image candidates for '{phrase1}' even with padding. Cannot proceed with AI selection.")
                     return

                best_image_url_1, selection_reasoning_1 = None, None
                with st.spinner(f"Asking AI to select the best image for '{phrase1}'..."):
                    best_image_url_1, selection_reasoning_1 = image_selector.choose_best_image(text_input, candidate_image_urls_1[:5]) # Pass exactly 5 URLs

                if best_image_url_1:
                    st.success(f"AI selected the best image for '{phrase1}'!")
                    st.info("AI Selection Reasoning for Image 1:")
                    st.write(selection_reasoning_1)
                    st.write(f"Selected Image 1 URL: {best_image_url_1}")
                else:
                    st.error(f"AI failed to select a suitable image for '{phrase1}'.")
                    return # Stop if AI selection fails for the first image

                # Step 5: Select the Best Image for Phrase 2 using AI
                st.subheader(f"Step 5: Selecting the Best Image for '{phrase2}' with AI")

                # Ensure we have at least 5 candidates for GPT-4 input (pad if necessary)
                while len(candidate_image_urls_2) < 5 and candidate_image_urls_2:
                     candidate_image_urls_2.append(candidate_image_urls_2[0]) # Pad with duplicates
                while len(candidate_image_urls_2) < 5:
                     # This case should ideally not be reached
                     st.error(f"Insufficient image candidates for '{phrase2}' even with padding. Cannot proceed with AI selection.")
                     return

                best_image_url_2, selection_reasoning_2 = None, None
                with st.spinner(f"Asking AI to select the best image for '{phrase2}'..."):
                    best_image_url_2, selection_reasoning_2 = image_selector.choose_best_image(text_input, candidate_image_urls_2[:5]) # Pass exactly 5 URLs

                if best_image_url_2:
                    st.success(f"AI selected the best image for '{phrase2}'!")
                    st.info("AI Selection Reasoning for Image 2:")
                    st.write(selection_reasoning_2)
                    st.write(f"Selected Image 2 URL: {best_image_url_2}")
                else:
                    st.error(f"AI failed to select a suitable image for '{phrase2}'.")
                    return # Stop if AI selection fails for the second image

                # --- Display Selected Images Separately ---
                st.subheader("Step 6: Selected Images")
                st.image(best_image_url_1, caption="Selected Image 1", use_column_width=True)
                st.image(best_image_url_2, caption="Selected Image 2", use_column_width=True)
                st.write("Review the selected images above before proceeding to the collage.")
                # ------------------------------------------

                # Step 6: Download Selected Images
                st.subheader("Step 7: Downloading Selected Images")
                progress_bar = st.progress(0)
                
                st.write("📥 Downloading first selected image...")
                image_data1 = download_image(best_image_url_1)
                progress_bar.progress(50)

                st.write("📥 Downloading second selected image...")
                image_data2 = download_image(best_image_url_2)
                progress_bar.progress(100)

                # Step 7: Create Collage
                if image_data1 and image_data2:
                    st.subheader("Step 8: Creating Collage")
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
                    st.error("Failed to download one or both selected images. Cannot create collage.")

            else:
                st.error("Failed to generate search phrases. Please try again.")

if __name__ == "__main__":
    main() 