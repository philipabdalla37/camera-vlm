import openai
import base64
import os
import tkinter as tk
from openai import OpenAI
from dotenv import load_dotenv
from Constants import *

#Prompt for GPT-4o
PROMPT = "Identify the number on the top face of the 6-sided die, even if the die is rotated or tilted. Answer with a single digit only. Do not confuse number 2 with number 5."

# Initialize OpenAI client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def SendToGpt(image=PROCESSED_IMAGE):
    try:
        # Read and encode the image
        with open(image, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Format as data URL
        image_data_url = f"data:image/jpeg;base64,{base64_image}"

        # Ask GPT-4o to detect the die number
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    ]
                }
            ]
        )

        # Extract GPT's response
        reply = response.choices[0].message.content
        print("GPT-4o says:", reply)
        return f'GPT-4o says: {reply}'
    
    except Exception as e:
        return f"An unexpected error occurred: {e}"