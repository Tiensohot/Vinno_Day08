import os
from dotenv import load_dotenv
from google import genai


load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
print("GOOGLE_API_KEY =", repr(API_KEY))

if not API_KEY:
    raise SystemExit("Chưa thấy biến môi trường GOOGLE_API_KEY")

client = genai.Client(api_key=API_KEY)

while True:
    msg = input("Bạn: ").strip()
    if msg.lower() in {"exit", "quit", "/exit"}:
        break

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=msg,
    )
    print("Gemini:", response.text)