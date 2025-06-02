# Text-to-Image Generator

A powerful web application that transforms text content into stunning AI-generated images. This tool uses OpenRouter's API for text analysis and Apify for high-quality image generation.

## Features

- 🎯 **Smart Text Analysis**: Uses OpenRouter's API to understand and interpret text content
- 🎨 **High-Quality Image Generation**: Creates detailed, realistic images using Apify
- 💻 **User-Friendly Interface**: Clean and intuitive Streamlit-based web interface
- ⚡ **Real-time Processing**: Quick response times for text analysis and image generation
- 🛡️ **Error Handling**: Robust error handling for API limits and connection issues
- 🔒 **Secure**: API key management through environment variables

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
   OPENROUTER_KEY=your_openrouter_key_here
   ```

4. Run the application:
   ```bash
   source venv/bin/activate
   python3 -m streamlit run app.py
   ```

## Usage

1. Launch the application using the command above
2. Enter or paste your text in the input area
3. Click "Generate Image"
4. Wait for the image generation process to complete
5. View and download your generated image

## Requirements

- Python 3.8 or higher
- Internet connection
- Required Python packages (automatically installed via requirements.txt):
  - streamlit==1.32.0
  - python-dotenv==1.0.1
  - Pillow==10.2.0
  - apify-client
  - openrouter

## License

This project is licensed under the MIT License - see the LICENSE file for details.

 