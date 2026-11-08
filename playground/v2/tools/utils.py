from pathlib import Path

WORKSPACE = Path(__file__).parent / "workspace"

def list_files() -> str:
    """List all files in the assistant's workspace folder."""
    if not WORKSPACE.exists() or not WORKSPACE.is_dir():
        return "Workspace folder does not exist."

    return "\n".join([str(file) for file in WORKSPACE.iterdir() if file.is_file()])

def read_file(file_name: str) -> str:
    """Read the content of a file in the assistant's workspace folder."""
    file_path = WORKSPACE / file_name
    if not file_path.exists() or not file_path.is_file():
        return f"File '{file_name}' does not exist."

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file '{file_name}': {str(e)}"

def write_file(file_name: str, content: str) -> str:
    """Write or overwrite the content of a file in the assistant's workspace folder."""
    file_path = WORKSPACE / file_name
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        return f"File '{file_name}' has been written successfully."
    except Exception as e:
        return f"Error writing to file '{file_name}': {str(e)}"
    

TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
}

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in the assistant's workspace folder.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a file in the assistant's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "The name of the file to read.",
                    }
                },
                "required": ["file_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite the content of a file in the assistant's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "The name of the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file.",
                    }
                },
                "required": ["file_name", "content"],
            },
        },
    },
]