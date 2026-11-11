from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import re
from tools.utils import TOOLS, TOOL_SCHEMA, load_memory, save_memory, list_files, read_file, write_file

load_dotenv(override=True)

# This is a simple example of an agent that can call tools and then send the results back to the model for further processing. 
# The agent will keep calling tools until the model returns a response without any tool calls.
THINKING = {"extra_body": {"google": {"thinking_config": {"thinking_level": "low", "include_thoughts": True}}}}
def split_thought(content):
    """Split the model's response into thought and answer parts."""
    content = content or ""
    thought = "\n".join(re.findall(r"<thought>(.*?)</thought>", content, re.S)).strip()
    answer = re.sub(r"<thought>.*?</thought>", "", content, flags=re.S).strip()
    return thought, answer


BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = "gemini-3.1-flash-lite"
API_KEY = os.getenv("GEMINI_API_KEY")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are a helpful personal assistant. 

Here is what you remember about the user from previous conversations:
{memory}

Once user asks you; first summrize the question in your own words, add it to the memory using the save_memory tool, then answer the question.
Once you answer the question, you must also add the answer to the memory using the save_memory tool.
If you need to read or write files in your workspace, use the tools provided.
You must first check the tools to you to be aware of what you can do. 
"""

def chat(messages: list) -> str:
    
    # The agent loop. This is the whole trick to make the agent call tools and then send the results back to the model for further processing.
    while True:
        response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOL_SCHEMA,
        extra_body=THINKING
        )

        message = response.choices[0].message
        thought, answer = split_thought(message.content)
        print("=" * 30)
        print(f"THINKING: {thought or '(none returned)'}")
        print("=" * 30)

        # Check if the response contains a tool call
        if not message.tool_calls:
            print("No tool call detected. Returning model response.")
            return answer

        
        # keep the assistant's tool-call message (incl. thought_signature) before the tool results
        messages.append(message)

        # if model suggests a tool call
        for call in message.tool_calls:
            tool_name = call.function.name
            tool_args = json.loads(call.function.arguments or "{}")
            print(f"Tool call detected: {tool_name} with arguments {tool_args}")
            
            result = TOOLS[tool_name](**tool_args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    


if __name__ == '__main__':
    print(f"v4 Assistant - ctrl + c to exit")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(memory=load_memory())},
    ]
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
            messages.append({"role": "user", "content": user_input})
            assistant_response = chat(messages)
            print(f"Assistant: {assistant_response}")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")