# chat.py (updated for openai>=1.0.0)

import os
from dotenv import load_dotenv

import openai
from openai import OpenAI  # <-- new import for the v1.x client

# ── 1. Load environment variables from .env ─────────────────────────────────
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env")

# ── 2. Instantiate the v1.x OpenAI client ─────────────────────────────────
# You can pass the key explicitly, or rely on the environment variable.
# Here, we pass it directly for clarity:
client = OpenAI(api_key=OPENAI_KEY)

# ── 3. Define a function to send a prompt and get a response ────────────────
def send_prompt_to_openai(prompt_text: str, model_name: str = "gpt-3.5-turbo") -> str:
    """
    Sends `prompt_text` to the specified OpenAI model (default: gpt-3.5-turbo)
    and returns the assistant’s reply as a string.
    """
    try:
        # In v1.x, call through client.chat.completions.create(...)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=150,
            temperature=0.7
        )
        # Extract the assistant’s reply
        answer = response.choices[0].message.content.strip()
        return answer

    except Exception as e:
        return f"Error calling OpenAI API: {e}"

# ── 4. If run directly, prompt for user input and print the response ───────
if __name__ == "__main__":
    print("=== Simple OpenAI Prompt Sender (v1.x) ===")
    user_prompt = input("Enter your prompt: ").strip()
    if not user_prompt:
        print("No prompt provided. Exiting.")
    else:
        reply = send_prompt_to_openai(user_prompt)
        print("\n--- OpenAI Response ---\n")
        print(reply)
