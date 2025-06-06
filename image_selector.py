import os
import json
import requests
import base64
from typing import List, Optional, Tuple
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
import re

load_dotenv()

class ImageSelector:
    def __init__(self, model_name: str = "gpt-4-vision-preview"):
        # Use OpenRouter API key
        self.openrouter_key = os.getenv('OPENROUTER_API_KEY')
        if not self.openrouter_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        self.model_name = model_name
        
        # Use OpenRouter API endpoint
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "HTTP-Referer": "https://github.com/yourusername/textoimg",
            "Content-Type": "application/json"
        }

    def _validate_and_process_image(self, image_data: bytes, content_type: str) -> Optional[str]:
        """
        Validate and process image data to ensure it's compatible with the vision model.
        
        Args:
            image_data (bytes): Raw image data
            content_type (str): MIME type of the image
            
        Returns:
            Optional[str]: Processed base64 string or None if validation fails
        """
        try:
            # Open image with PIL to validate and process
            img = Image.open(BytesIO(image_data))
            
            # Convert to RGB if necessary (handles RGBA, etc.)
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            
            # Resize if too large (vision models have size limits)
            MAX_SIZE = 1024
            if max(img.size) > MAX_SIZE:
                ratio = MAX_SIZE / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to bytes
            output = BytesIO()
            img.save(output, format='JPEG', quality=95)
            processed_data = output.getvalue()
            
            # Convert to base64
            base64_string = base64.b64encode(processed_data).decode('utf-8')
            
            return base64_string
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def _download_and_encode_image(self, image_url: str) -> Optional[str]:
        """
        Download an image from URL and convert it to base64 string.
        
        Args:
            image_url (str): URL of the image to download
            
        Returns:
            Optional[str]: Base64 encoded image string, or None if failed
        """
        try:
            # Add more headers to mimic a browser request more closely
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/webp,image/apng,image/svg+xml,image/*,;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.google.com/' # Add a referer header as well
            }
            
            # Download the image with a timeout and headers
            response = requests.get(image_url, timeout=10, headers=headers)
            response.raise_for_status()
            
            # Get content type from response headers
            content_type = response.headers.get('content-type', 'image/jpeg')
            
            # Validate content type
            if not content_type.startswith('image/'):
                print(f"Warning: Invalid content type {content_type} for {image_url}")
                return None
            
            # Process and validate the image
            return self._validate_and_process_image(response.content, content_type)
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image from {image_url}: {str(e)}")
            return None
        except Exception as e:
            print(f"Error encoding image from {image_url}: {str(e)}")
            return None

    def choose_best_image(self, summary: str, image_urls: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Select the best image from the provided URLs based on the article summary using the vision model.
        Returns the URL of the best image and the reasoning from the model.
        
        Args:
            summary (str): The summarized text of the article
            image_urls (List[str]): List of image URLs to choose from (will use up to 5)
            
        Returns:
            Tuple[Optional[str], Optional[str]]: A tuple containing the URL of the best image and the reasoning, or (None, None) if selection fails.
        """
        # Ensure exactly 5 image URLs for the vision model API input
        processed_image_urls = []
        if len(image_urls) == 0:
            print("Error: No image URLs provided to ImageSelector.")
            return (None, "No image candidates found.")
        
        if len(image_urls) < 5:
             print(f"Warning: Expected 5 image URLs, received {len(image_urls)}. Padding with duplicates.")
             processed_image_urls = list(image_urls) # Create a mutable copy
             while len(processed_image_urls) < 5:
                 processed_image_urls.append(processed_image_urls[0]) # Pad with duplicates of the first image
        elif len(image_urls) > 5:
             print(f"Warning: Expected 5 image URLs, received {len(image_urls)}. Using the first 5.")
             processed_image_urls = image_urls[:5] # Use only the first 5
        else:
            processed_image_urls = list(image_urls) # Use the list as is if already 5

        try:
            # Encode images to base64
            encoded_images = []
            for img_url in processed_image_urls:
                base64_img = self._download_and_encode_image(img_url)
                if base64_img:
                    encoded_images.append(base64_img)
                else:
                    print(f"Warning: Failed to process image {img_url}, skipping...")

            if not encoded_images:
                return (None, "Failed to process any valid images for analysis.")

            # Construct the messages with images
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""You are an AI visual editor for a major news agency. Your task is to select the BEST image from the {len(encoded_images)} provided that most accurately and powerfully illustrates the following news summary.

News Summary:
{summary}

Your response MUST be in the following strict two-line format:
Line 1: The single number (1-{len(encoded_images)}) corresponding to the BEST image.
Line 2: A single sentence explaining your choice.

Example Response:
3
This image clearly shows the main subject discussed in the article.

DO NOT include any other text, formatting, or commentary outside of these two lines."""
                        }
                    ]
                }
            ]
            
            # Add each image to the messages with the new GPT-4 Vision format
            for encoded_img in encoded_images:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_img}"
                    }
                })
            
            data = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0 # Set temperature to 0 for more predictable output
            }
            
            # Make the API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data
            )
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Extract the model's response
            model_response = ""
            if 'choices' in result and len(result['choices']) > 0:
                if 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
                    model_response = result['choices'][0]['message']['content']
                else:
                    raise ValueError("Unexpected response format: missing message or content")
            else:
                raise ValueError(f"Unexpected response format from vision model: {json.dumps(result)}")
            
            print("\nRaw Model response:", model_response)
            
            # --- Improved Parsing Logic ---
            selected_index = None
            reasoning = "Error parsing AI response."

            lines = [line.strip() for line in model_response.split('\n') if line.strip()]
            
            if lines:
                # Try to get the index from the first non-empty line
                try:
                    potential_index = int(lines[0])
                    if 1 <= potential_index <= len(processed_image_urls):
                        selected_index = potential_index
                        # Get reasoning from the second line if available
                        if len(lines) > 1:
                            reasoning = lines[1]
                        else:
                             reasoning = "No reasoning provided by AI."
                except ValueError:
                    # If the first line is not an integer, try to find a number elsewhere
                    numbers = re.findall(r'\b\d+\b', model_response)
                    if numbers:
                        potential_index = int(numbers[0])
                        if 1 <= potential_index <= len(processed_image_urls):
                             selected_index = potential_index
                             # If index found later, use the whole response as reasoning
                             reasoning = model_response.strip()

            # --- End of Improved Parsing Logic ---

            if selected_index is not None:
                 selected_image_url = processed_image_urls[selected_index - 1]
                 print(f"AI selected image index: {selected_index}")
                 return selected_image_url, reasoning
            else:
                print("Failed to parse a valid image selection from the AI response.")
                return (None, "Failed to parse AI response.")
                
        except requests.exceptions.RequestException as e:
            print(f"API request error: {str(e)}")
            return (None, f"API request error: {str(e)}")
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            return (None, f"JSON parsing error: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return (None, f"Unexpected error: {str(e)}") 