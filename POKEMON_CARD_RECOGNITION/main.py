import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

# 1. Load environment variables from .env (especially the API key).
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. Helper: read the image file and encode it in base64.
def image_to_base64(path):
    # Ensure the file exists, otherwise open() will error.
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# 3. Function that sends the image + question to GPT-4V.
def ask_about_image(image_path, question):
    # Convert the local image into a base64 data URI
    image_b64 = image_to_base64(image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "אתה מדריך חכם, נחמד וסבלני שמסביר לילדים בני 6 על קלפים "
                    "ממשחקי קלפים כמו פוקימון. המטרה שלך היא להסביר בעברית "
                    "פשוטה מה הקלף עושה, איך משתמשים בו במשחק, ולתת דוגמה "
                    "אחת או שתיים לשימוש בכיף – כאילו אתה מספר סיפור קטן. אל "
                    "תשתמש במילים מסובכות או מושגים טכניים. דבר בטון חמוד ומעודד, "
                    "ותמיד תסיים עם חיוך קטן או שאלה שמעודדת את הילד ללמוד עוד."
                ),
            },
            {
                "role": "user",
                # We send a list of two objects: the text question, and the image payload:
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                            # The `gpt-4-vision-preview` endpoint accepts base64-encoded data URIs.
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        },
                    },
                ],
            },
        ],
        max_tokens=500,
    )
    # Return just the assistant’s textual reply (stripped of leading/trailing whitespace).
    return response.choices[0].message.content.strip()

# 4. This ensures the code below only runs when you call `python ask_card_bot.py`.
if __name__ == "__main__":
    # Make sure this path points to a real image in the same folder,
    # or give an absolute path.
    image_path = "./WhatsApp Image 2025-05-31 at 18.40.33.jpeg"
    question = "היי מה זה עושה?"
    reply = ask_about_image(image_path, question)
    print("GPT עונה:\n", reply)
