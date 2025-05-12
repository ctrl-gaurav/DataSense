from dotenv import load_dotenv
import os
from pathlib import Path
from google import genai

def get_gemini_client():
    # Load environment variables from .env file
    load_dotenv()
    # Get the API key
    api_key = os.getenv("GEMINI_KEY")
    client = genai.Client(api_key=api_key)
    return client

# Function to replace placeholder in the .md file
def replace_placeholder_in_md(file_path, placeholder_and_value):
    # Open the markdown file and read its content
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Replace the placeholder with the dynamic value
    for k, v in placeholder_and_value.items():
        content = content.replace(k, v)

    return content