# Text-to-Image Generator

A powerful web application that transforms text content into stunning AI-generated images. This tool uses OpenAI's GPT-4 to analyze and interpret text content, then leverages DALL·E 3 to create high-quality, contextually relevant images that match the text description.

## Features

- 🎯 **Smart Text Analysis**: Uses GPT-4 to understand and interpret text content
- 🎨 **High-Quality Image Generation**: Creates detailed, realistic images using DALL·E 3
- 💻 **User-Friendly Interface**: Clean and intuitive Streamlit-based web interface
- ⚡ **Real-time Processing**: Quick response times for text analysis and image generation
- 🛡️ **Error Handling**: Robust error handling for API limits and connection issues
- 🔒 **Secure**: API key management through environment variables

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/textoimg.git
   cd textoimg
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   Apify and Openrouter keys required, just paste like this: 
   APIFY_TOKEN= *key*
   OPENROUTER_KEY= *key*
   ```

4. Run the application:
   ```bash
   streamlit run app.py '''(or python3 -m streamlit run app.py)'''
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
  - openai==0.28.1
  - python-dotenv==1.0.1
  - streamlit==1.32.0
  - Pillow==10.2.0

 
## License

This project is licensed under the MIT License - see the LICENSE file for details.

 