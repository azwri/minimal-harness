from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def chat(user_message: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print(f"v0 Assistant - ctrl + c to exit")
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            assistant_response = chat(user_input)
            print(f"Assistant: {assistant_response}")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")