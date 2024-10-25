import streamlit as st
from anthropic import Anthropic
import json
from datetime import datetime
import os
from duckduckgo_search import DDGS
from serpapi import GoogleSearch
import tiktoken
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Anthropic client - allow for both secrets and env vars
api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error(
        "No API key found. Please set ANTHROPIC_API_KEY in .env file or Streamlit secrets."
    )
    st.stop()

anthropic = Anthropic(api_key=api_key)

# Initialize tokenizer for token counting
tokenizer = tiktoken.encoding_for_model("gpt-4")


class MemoryManager:
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.memories: List[Dict] = []
        self.load_memories()

    def add_memory(self, content: str, importance: float = 1.0):
        """Add a new memory with timestamp and importance score"""
        memory = {
            "id": len(self.memories),
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "importance": importance,
            "tokens": len(tokenizer.encode(content)),
            "votes": 0,
        }
        self.memories.append(memory)
        self.save_memories()

    def update_memory_importance(self, memory_id: int, vote: int):
        """Update memory importance based on votes"""
        for memory in self.memories:
            if memory["id"] == memory_id:
                memory["votes"] += vote
                # Adjust importance based on votes (normalized between 0.5 and 2.0)
                memory["importance"] = max(0.5, min(2.0, 1.0 + (memory["votes"] * 0.1)))
                break
        self.save_memories()

    def get_relevant_memories(self, query: str, max_tokens: int = 2000) -> str:
        """Get relevant memories that fit within token budget"""
        if not self.memories:
            return ""

        df = pd.DataFrame(self.memories)
        if df.empty:
            return ""

        # Convert timestamps safely
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")

        # Calculate recency score with safe division
        time_range = (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
        if time_range > 0:
            df["recency_score"] = (
                df["timestamp"] - df["timestamp"].min()
            ).dt.total_seconds() / time_range
        else:
            df["recency_score"] = 1.0

        df["combined_score"] = df["importance"] * 0.7 + df["recency_score"] * 0.3
        df = df.sort_values("combined_score", ascending=False)

        selected_memories = []
        total_tokens = 0

        for _, memory in df.iterrows():
            if total_tokens + memory["tokens"] <= max_tokens:
                selected_memories.append(memory["content"])
                total_tokens += memory["tokens"]
            else:
                break

        return "\n\n".join(selected_memories)

    def save_memories(self):
        """Save memories to file"""
        with open("memories.json", "w") as f:
            json.dump(self.memories, f)

    def load_memories(self):
        """Load memories from file"""
        try:
            with open("memories.json", "r") as f:
                self.memories = json.load(f)
        except FileNotFoundError:
            self.memories = []

    def clear_memories(self):
        """Clear all memories"""
        self.memories = []
        if os.path.exists("memories.json"):
            os.remove("memories.json")


def get_search_tools():
    """Get the available search tools based on configuration"""
    tools = [{
        "name": "duckduckgo_search",
        "description": "Search the web using DuckDuckGo. Use this to find current information about topics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 3)",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    }]
    
    if os.getenv("SERPAPI_KEY"):
        tools.append({
            "name": "serpapi_search",
            "description": "Search the web using Google (via SerpAPI). Use this to find current information about topics.",
            "input_schema": {
                "type": "object", 
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 3)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        })
    
    return tools

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute the requested tool with the given input"""
    if tool_name == "duckduckgo_search":
        try:
            num_results = tool_input.get("num_results", 3)
            with DDGS() as ddgs:
                results = list(ddgs.text(tool_input["query"], max_results=num_results))

            if not results:
                return "No search results found."

            formatted_results = []
            for result in results:
                title = result.get("title", "No title available")
                link = result.get("link", "No link available") 
                snippet = result.get("body", "No snippet available")

                formatted_results.append(
                    f"Title: {title}\n"
                    f"Link: {link}\n"
                    f"Snippet: {snippet}\n"
                )

            return "\n\n".join(formatted_results)
        except Exception as e:
            st.error(f"DuckDuckGo search error: {str(e)}")
            return "Unable to perform web search at this time."

    elif tool_name == "serpapi_search":
        try:
            num_results = tool_input.get("num_results", 3)
            params = {
                "q": tool_input["query"],
                "num": num_results,
                "api_key": os.getenv("SERPAPI_KEY")
            }
            search = GoogleSearch(params)
            results = search.get_dict().get("organic_results", [])

            if not results:
                return "No search results found."

            formatted_results = []
            for result in results[:num_results]:
                title = result.get("title", "No title available")
                link = result.get("link", "No link available")
                snippet = result.get("snippet", "No snippet available")

                formatted_results.append(
                    f"Title: {title}\n"
                    f"Link: {link}\n"
                    f"Snippet: {snippet}\n"
                )

            return "\n\n".join(formatted_results)
        except Exception as e:
            st.error(f"SerpAPI search error: {str(e)}")
            return "Unable to perform web search at this time."


# Initialize memory manager
if "memory_manager" not in st.session_state:
    st.session_state.memory_manager = MemoryManager()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


def display_memories():
    """Display memories with voting buttons"""
    st.subheader("Memory Bank")

    if not st.session_state.memory_manager.memories:
        st.write("No memories stored yet.")
        return

    for memory in sorted(
        st.session_state.memory_manager.memories,
        key=lambda x: x["timestamp"],
        reverse=True,
    ):
        with st.expander(
            f"Memory {memory['id']} (Importance: {memory['importance']:.2f})"
        ):
            st.write(memory["content"])
            st.write(f"Created: {memory['timestamp']}")
            st.write(f"Votes: {memory['votes']}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍", key=f"upvote_{memory['id']}"):
                    st.session_state.memory_manager.update_memory_importance(
                        memory["id"], 1
                    )
                    st.rerun()  # Updated from experimental_rerun()
            with col2:
                if st.button("👎", key=f"downvote_{memory['id']}"):
                    st.session_state.memory_manager.update_memory_importance(
                        memory["id"], -1
                    )
                    st.rerun()  # Updated from experimental_rerun()


def load_chat_history():
    """Load chat history from file if it exists"""
    try:
        with open("chat_history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_chat_history(messages):
    """Save chat history to file"""
    with open("chat_history.json", "w") as f:
        json.dump(messages, f)


def get_assistant_response(
    prompt: str, history: List[Dict], include_web_search: bool = True
):
    """Get response from Claude API with memory and web search integration"""

    # Get relevant memories
    relevant_memories = st.session_state.memory_manager.get_relevant_memories(prompt)

    # Construct system message with context
    system_message = """You are a helpful AI assistant with access to memory and web search capabilities.
    When answering questions that require current information, use the search tools available to you.
    Always cite the sources from search results when you use them in your response.
    Please provide informative responses while maintaining a natural conversational style."""

    if relevant_memories:
        system_message += (
            "\n\nRelevant memories from previous conversations:\n" + relevant_memories
        )

    # Format messages for the API
    messages = []

    # Add historical context
    for msg in history:
        role = "assistant" if msg["role"] == "assistant" else "user"
        messages.append({"role": role, "content": msg["content"]})

    # Add current prompt
    messages.append({"role": "user", "content": prompt})

    try:
        # Get available search tools
        tools = get_search_tools() if include_web_search else []
        
        # Make initial request to Claude
        response = anthropic.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=4000,
            system=system_message,
            messages=messages,
            tools=tools,
            temperature=0.5,
        )

        # Handle tool use if needed
        while response.stop_reason == "tool_use":
            tool_use = response.content[0].text if isinstance(response.content[0].text, str) else response.content[-1]
            if not isinstance(tool_use, dict) or "tool_use" not in tool_use:
                break
                
            tool_result = execute_tool(tool_use["tool_use"]["name"], tool_use["tool_use"]["input"])
            
            # Add tool result to messages
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use["tool_use"]["id"],
                    "content": tool_result
                }]
            })
            
            # Get next response from Claude
            response = anthropic.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=4000,
                system=system_message,
                messages=messages,
                tools=tools,
                temperature=0.5,
            )

        final_response = response.content[0].text

        # Store important information in memory
        st.session_state.memory_manager.add_memory(
            f"User asked: {prompt}\nAssistant responded: {final_response}",
            importance=1.0,
        )

        return final_response
    except Exception as e:
        return f"Error: {str(e)}"


# [Previous imports and class definitions remain the same until the Page config section]

# Page config
st.set_page_config(page_title="Claude Chat", page_icon="🤖", layout="wide")

# Initialize session state variables
if "include_web_search" not in st.session_state:
    st.session_state.include_web_search = True
if "search_provider" not in st.session_state:
    st.session_state.search_provider = "serpapi" if os.getenv("SERPAPI_KEY") else "duckduckgo"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Main chat interface - Title at the top
st.title("Claude Chat Assistant")

# Chat input must be at root level
prompt = st.chat_input("What would you like to know?")

# Create two columns for chat history and memory display
col1, col2 = st.columns([2, 1])

with col1:
    # Load existing chat history
    if not st.session_state.messages:
        st.session_state.messages = load_chat_history()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process the prompt if it exists
    if prompt:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_assistant_response(
                    prompt,
                    st.session_state.messages[:-1],
                    st.session_state.include_web_search,
                )
                st.markdown(response)

        # Add assistant response to state
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Save updated chat history
        save_chat_history(st.session_state.messages)

with col2:
    # Settings and controls
    st.title("Settings")

    # Search settings
    st.session_state.include_web_search = st.checkbox(
        "Enable Web Search", value=st.session_state.include_web_search
    )
    
    search_options = ["duckduckgo"]
    if os.getenv("SERPAPI_KEY"):
        search_options.insert(0, "serpapi")
    
    st.session_state.search_provider = st.selectbox(
        "Search Provider",
        options=search_options,
        index=search_options.index(st.session_state.search_provider)
    )

    st.subheader("Memory Management")
    if st.button("Clear All Memories"):
        st.session_state.memory_manager.clear_memories()
        st.success("Memories cleared!")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        if os.path.exists("chat_history.json"):
            os.remove("chat_history.json")
        st.rerun()

    # Display memories with voting
    display_memories()

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Claude API, and DuckDuckGo Search")
