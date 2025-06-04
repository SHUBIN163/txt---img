# Text-to-Image Generator

A powerful web application that transforms text content into stunning AI-generated images. This tool uses OpenRouter's API for text analysis and **vision-based image selection**, and Apify for high-quality image generation.

## Features

- 🎯 **Smart Text Analysis**: Uses OpenRouter's API (including models like GPT-4) to understand and interpret text content and generate search phrases.
- 👀 **Vision-Based Image Selection**: Employs the `meta-llama/llama-3.2-90b-vision-instruct` model via OpenRouter to select the most relevant images from search results.
- 🎨 **High-Quality Image Generation**: Searches for relevant images using Apify's Google Images Scraper and processes them.
- ✨ **Collage Creation**: Combines the two selected images into a complementary collage.
- 💻 **User-Friendly Interface**: Clean and intuitive Streamlit-based web interface
- ⚡ **Real-time Processing**: Quick response times for text analysis and image generation
- 🛡️ **Robust Error Handling**: Improved error handling for API calls, image downloading, and parsing
- 🔒 **Secure API Key Management**: API keys managed through environment variables

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/SHUBIN163/txt---img.git
   cd txt---img
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   APIFY_TOKEN=your_apify_token_here
   OPENROUTER_API_KEY=your_openrouter_key_here
   ```

4. Run the application:
   ```bash
   source venv/bin/activate
   streamlit run app.py
   ```

## Usage

1. Launch the application using the command above
2. Enter or paste your text in the input area
3. Click "Generate Images"
4. The application will generate search phrases, find image candidates, use AI to select two best images, download them, and create a collage
5. View the selected images and the final collage in the web interface
6. You can download the generated collage

## Requirements

- Python 3.8 or higher
- Internet connection
- Required Python packages (automatically installed via requirements.txt):
  - streamlit
  - python-dotenv
  - Pillow
  - apify-client
  - openrouter
  - requests

## License

This project is licensed under the MIT License - see the LICENSE file for details.

 