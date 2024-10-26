import json
from typing import List, Dict

def load_chat_history() -> List[Dict]:
    """Load chat history from file if it exists"""
    try:
        with open("chat_history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_chat_history(messages: List[Dict]) -> None:
    """Save chat history to file"""
    with open("chat_history.json", "w") as f:
        json.dump(messages, f)
