import streamlit as st
from anthropic import Anthropic
import json
from datetime import datetime
import os
from search.providers.duckduckgo import execute_duckduckgo_search
from search.providers.serpapi import execute_serpapi_search
from search.providers.firecrawl import execute_firecrawl
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv
from memory import MemoryManager, ToolUsageCounter
import tiktoken

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



from search import get_search_tools, execute_tool

def get_available_tools():
    """Get all available tools based on configuration"""
    tools = [
        {
            "name": "memory_manager", 
            "description": """Manage long-term memory storage. Store new memories, update existing ones, or search memories.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["store", "update"],
                        "description": "Action to perform on memories",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to store or search for",
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 2.0,
                        "description": "Importance rating (0.0-2.0)",
                    },
                    "memory_id": {
                        "type": "integer",
                        "description": "ID of memory to update (for update action)",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for importance rating or update",
                    },
                },
                "required": ["action", "content"],
            },
        },
        {
            "name": "visit",
            "description": """Read a webpage's content. Returns clean, readable text.
            Optional extraction guidance to efficiently grab specific details.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Webpage URL to read"},
                    "extract": {
                        "type": "string",
                        "description": "Describe the specific information to extract from the webpage",
                    },
                },
                "required": ["url"],
            },
        },
    ]

    # Add the selected search provider
    if st.session_state.search_provider == "serpapi":
        tools.append(
            {
                "name": "google_search",
                "description": "Search Google for current information.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to send to Google. Should be specific and focused on the information needed.",
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of search results to return (1-15, default 5)",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 15,
                        },
                    },
                    "required": ["query"],
                },
            }
        )
    elif st.session_state.search_provider == "duckduckgo":
        tools.append(
            {
                "name": "search",
                "description": """Find information across the web. Returns titles, snippets and links. Use to research real-time and up-to-date information from anywhere on the internet.""",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to send to DuckDuckGo. Should be specific and focused on the information needed.",
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of search results to return (1-15, default 5)",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 15,
                        },
                    },
                    "required": ["query"],
                },
            }
        )

    return tools




def execute_tool(tool_name: str, tool_args: dict) -> str:
    """Execute the requested tool with the given input"""
    try:
        if tool_name == "memory_manager":
            action = tool_args.get("action")
            content = tool_args.get("content")

            if action == "store":
                importance = tool_args.get("importance", 1.0)
                reason = tool_args.get("reason", "")
                st.session_state.memory_manager.add_memory(
                    content, importance=importance, metadata={"reason": reason}
                )
                return f"Memory stored with importance {importance}"

            elif action == "update":
                memory_id = tool_args.get("memory_id")
                importance = tool_args.get("importance")
                if memory_id is None:
                    return {
                        "error": "memory_id required for update action",
                        "is_error": True,
                    }

                for memory in st.session_state.memory_manager.memories:
                    if memory["id"] == memory_id:
                        memory["content"] = content
                        if importance:
                            memory["importance"] = importance
                        st.session_state.memory_manager.save_memories()
                        return f"Memory {memory_id} updated"
                return f"Memory {memory_id} not found"

            return {"error": f"Unknown memory action: {action}", "is_error": True}

        elif tool_name == "visit":
            if "url" not in tool_args:
                return {"error": "Missing required 'url' parameter", "is_error": True}
            return execute_firecrawl(tool_args["url"], tool_args.get("prompt"))

        elif tool_name in ["search", "google_search"]:
            if "query" not in tool_args:
                return {"error": "Missing required 'query' parameter", "is_error": True}

            # Get and validate num_results
            num_results = tool_args.get("num_results", 5)
            num_results = min(max(num_results, 1), 15)
            query = tool_args["query"]

            if tool_name == "search":
                try:
                    return execute_duckduckgo_search(query, num_results)
                except Exception as e:
                    return {
                        "error": f"DuckDuckGo search failed: {str(e)}",
                        "is_error": True,
                    }
            elif tool_name == "google_search":
                try:
                    return execute_serpapi_search(query, num_results)
                except Exception as e:
                    return {
                        "error": f"SerpAPI search failed: {str(e)}",
                        "is_error": True,
                    }
        else:
            return {"error": f"Unknown tool '{tool_name}'", "is_error": True}

    except Exception as e:
        st.error(f"Tool execution error: {str(e)}")
        return {"error": f"Error executing {tool_name}: {str(e)}", "is_error": True}


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


from search.chat import load_chat_history, save_chat_history, get_assistant_response
from search.search.tools import get_available_tools



# [Previous imports and class definitions remain the same until the Page config section]

# Page config
st.set_page_config(page_title="Claude Chat", page_icon="🤖", layout="wide")

# Initialize session state variables
if "progress_updates" not in st.session_state:
    st.session_state.progress_updates = []
if "include_web_search" not in st.session_state:
    st.session_state.include_web_search = True
if "search_provider" not in st.session_state:
    st.session_state.search_provider = (
        "serpapi" if os.getenv("SERPAPI_KEY") else "duckduckgo"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Main chat interface - Title at the top
st.title("search.tommckenzie.dev")

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
            response_placeholder = st.empty()
            with st.spinner("Thinking..."):
                # Add custom CSS for the scrollable container
                st.markdown(
                    """
                    <style>
                    .stExpander div[data-testid="stExpander"] {
                        max-height: 80dvh;
                        overflow-y: auto;
                    }
                    </style>
                """,
                    unsafe_allow_html=True,
                )

                progress_expander = st.expander("View Progress", expanded=False)
                with progress_expander:
                    summary_placeholder = st.empty()
                    progress_placeholder = st.empty()

                # Clear previous progress updates
                st.session_state.progress_updates = []
                # Initialize tool counter
                tool_counter = ToolUsageCounter()

                def progress_callback(action, details=None, counter=tool_counter):
                    update = {
                        "action": f"🔄 {action}",
                        "timestamp": datetime.now().isoformat(),
                    }
                    if details:
                        update["details"] = details
                    st.session_state.progress_updates.append(update)

                    # Display summary and updates
                    with progress_placeholder.container():
                        summary = tool_counter.get_summary()
                        summary_placeholder.markdown(f"**Summary:** {summary}")
                        st.markdown("---")
                        for update in st.session_state.progress_updates:
                            st.write(update["action"])
                            if "details" in update:
                                st.code(update["details"], language="json")

                response = get_assistant_response(
                    prompt,
                    st.session_state.messages[:-1],
                    st.session_state.include_web_search,
                    progress_callback,
                )
                response_placeholder.markdown(response)

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
        index=search_options.index(st.session_state.search_provider),
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
