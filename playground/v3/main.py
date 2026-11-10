import asyncio
from openai import AsyncOpenAI
from contextlib import AsyncExitStack
from dotenv import load_dotenv
import os
import json
import re
from tools.utils import list_files, read_file, TOOLS, TOOL_SCHEMA

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


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

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

MCP_SERVERS = {
    'time': StdioServerParameters(command='uvx', args=['mcp-server-time']),
    'fetch': StdioServerParameters(command='uvx', args=['mcp-server-fetch'])
}



# --- NEW: the MCP component -------------------------------------------------

# which MCP server owns each remote tool (tool name -> live session)
mcp_sessions: dict[str, ClientSession] = {}

async def connect_mcp(stack: AsyncExitStack):
    """Launch each MCP server and merge its tools into TOOL_SCHEMAS."""
    for name, params in MCP_SERVERS.items():
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        tools = (await session.list_tools()).tools
        for tool in tools:
            mcp_sessions[tool.name] = session
            TOOL_SCHEMA.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
            )
        print(f"  [mcp] connected '{name}': {[t.name for t in tools]}")


async def call_tool(name: str, args: dict) -> str:
    """Run a tool - ours directly, or an MCP server's over the protocol."""
    if name in TOOLS:
        return TOOLS[name](**args)
    result = await mcp_sessions[name].call_tool(name, args)
    return "\n".join(c.text for c in result.content if hasattr(c, "text"))



async def chat(user_message: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message},
    ]
    
    # The agent loop. This is the whole trick to make the agent call tools and then send the results back to the model for further processing.
    while True:
        response = await client.chat.completions.create(
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
            
            result = await call_tool(tool_name, tool_args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    


async def main():
    async with AsyncExitStack() as stack:
        await connect_mcp(stack)
        print(f"\nv3 assistant ({MODEL}) with {len(TOOL_SCHEMA)} tools - ctrl+c to quit")
        try:
            while True:
                question = await asyncio.to_thread(input, "\nyou: ")
                print("\nassistant:", await chat(question))
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")


if __name__ == "__main__":
    asyncio.run(main())