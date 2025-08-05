import os
import json
import requests
import asyncio
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

# To be used with the new async function
# The `requests` library is blocking. For a truly async solution, you would use an async library like `httpx`.
# For now, we'll wrap the blocking call with asyncio.to_thread.

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY is None:
    raise ValueError("Please set your GEMINI_API_KEY environment variable")

headers = {
    "Content-Type": "application/json"
}

def generate_answer_gemini(prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
    """
    Generates text using Google Gemini Inference API (blocking version).

    Args:
        prompt (str): The input prompt.
        max_tokens (int): Maximum new tokens to generate.
        temperature (float): Controls randomness in generation.

    Returns:
        str: Generated text from Gemini model.
    """

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }

    url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

    try:
        response = requests.post(url_with_key, headers=headers, json=payload)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error: {err}")
        print(f"Response Body: {response.text}")
        raise
    except requests.exceptions.RequestException as err:
        raise Exception(f"Request failed: {err}")

    res_json = response.json()

    try:
        candidates = res_json.get("candidates", [])
        if not candidates:
            feedback = res_json.get("promptFeedback")
            if feedback:
                raise Exception(f"Prompt blocked by safety settings: {feedback}")
            raise Exception("No candidates returned from Gemini API")
        
        generated_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        if not generated_text:
            raise Exception("Could not find generated text in Gemini response.")

        return generated_text
    
    except (KeyError, IndexError) as e:
        raise Exception(f"Failed to parse Gemini response: {e}. Full response: {json.dumps(res_json, indent=2)}")