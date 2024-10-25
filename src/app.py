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

    def add_memory(self, content: str, importance: float = 1.0, metadata: dict = None):
        """Add a new memory with timestamp, importance score and metadata"""
        memory = {
            "id": len(self.memories),
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "importance": importance,
            "tokens": len(tokenizer.encode(content)),
            "votes": 0,
            "metadata": metadata or {},
        }
        self.memories.append(memory)
        self.save_memories()

    def update_memory_importance(self, memory_id: int, vote: int):
        """Update memory importance based on votes"""
        for memory in self.memories:
            if memory["id"] == memory_id:
                memory["votes"] += vote
                # Adjust importance relative to current value (bounded between 0.5 and 2.0)
                memory["importance"] = max(
                    0.5, min(2.0, memory["importance"] + (vote * 0.1))
                )
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


def execute_duckduckgo_search(query: str, num_results: int) -> str:
    """Execute a DuckDuckGo search"""
    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    query,
                    max_results=num_results,
                    region="wt-wt",
                    safesearch="moderate",
                )
            )

        if not results:
            return "No search results found."

        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"[Result {i}]\n"
                f"Title: {result.get('title', 'No title')}\n"
                f"URL: {result.get('link') or result.get('url', 'No URL')}\n"
                f"Description: {result.get('body', 'No description')}\n"
                f"Published: {result.get('published', 'Date unknown')}\n"
            )

        return "\n\n".join(formatted_results)
    except Exception as e:
        st.error(f"DuckDuckGo search error: {str(e)}")
        return "Unable to perform DuckDuckGo search at this time."


def execute_serpapi_search(query: str, num_results: int) -> str:
    """Execute a Google search via SerpAPI"""
    try:
        params = {
            "q": query,
            "num": num_results,
            "api_key": os.getenv("SERPAPI_KEY"),
            "hl": "en",
            "gl": "us",
            "safe": "active",
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        if "error" in results:
            return f"SerpAPI error: {results['error']}"

        organic_results = results.get("organic_results", [])
        if not organic_results:
            return "No search results found."

        formatted_results = []
        for i, result in enumerate(organic_results[:num_results], 1):
            formatted_results.append(
                f"[Result {i}]\n"
                f"Title: {result.get('title', 'No title')}\n"
                f"URL: {result['link']}\n"
                f"Description: {result.get('snippet', 'No description')}\n"
                f"Position: {result.get('position', 'Unknown')}\n"
                f"Displayed URL: {result.get('displayed_link', 'No URL')}\n"
            )

            if "rich_snippet" in result:
                rich = result["rich_snippet"]
                if "top" in rich:
                    formatted_results[-1] += f"Additional Info: {rich['top']}\n"

        return "\n\n".join(formatted_results)
    except Exception as e:
        st.error(f"SerpAPI search error: {str(e)}")
        return "Unable to perform Google search at this time."


def execute_firecrawl(url: str, prompt: str = None) -> str:
    """Execute a Firecrawl web extraction"""
    try:
        from firecrawl import FirecrawlApp

        app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

        params = {"formats": ["markdown"]}

        if prompt:
            params["formats"].append("extract")
            params["extract"] = {"prompt": prompt}

        result = app.scrape_url(url, params=params)

        # Return markdown content by default
        content = result.get("markdown", "")

        # If extraction was requested, append it to the content
        if prompt and "extract" in result:
            content += "\n\nExtracted Information:\n" + str(result["extract"])

        return content
    except Exception as e:
        st.error(f"Firecrawl error: {str(e)}")
        return "Unable to extract webpage content at this time."


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


class ToolUsageCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.thought_count = 0
        self.search_count = 0
        self.visit_count = 0
        self.memory_count = 0

    def count_tool(self, tool_name: str):
        if tool_name in ["search", "google_search"]:
            self.search_count += 1
        elif tool_name == "visit":
            self.visit_count += 1
        self.thought_count += 1

    def set_memory_count(self, count: int):
        self.memory_count = count

    def get_summary(self) -> str:
        parts = []
        if self.thought_count > 0:
            parts.append(f"Thought {self.thought_count} times")
        if self.search_count > 0:
            parts.append(
                f"performed {self.search_count} search{'' if self.search_count == 1 else 'es'}"
            )
        if self.visit_count > 0:
            parts.append(
                f"made {self.visit_count} visit{'' if self.visit_count == 1 else 's'}"
            )
        if self.memory_count > 0:
            parts.append(
                f"used {self.memory_count} relevant memor{'y' if self.memory_count == 1 else 'ies'}"
            )

        if not parts:
            return "No tools used"

        return ", ".join(parts)


def get_assistant_response(
    prompt: str,
    history: List[Dict],
    include_web_search: bool = True,
    progress_callback=None,
):
    """Get response from Claude API with memory and web search integration"""
    # Get relevant memories and count them
    relevant_memories = st.session_state.memory_manager.get_relevant_memories(prompt)
    # Only count non-empty memory entries
    memory_count = (
        len([m for m in relevant_memories.split("\n\n") if m.strip()])
        if relevant_memories
        else 0
    )
    tool_counter.set_memory_count(memory_count)

    # Construct system message with context and chain-of-thought prompting
    system_message = """You are a focused and powerful researcher, with deep intuition and a careful eye for entity disambiguation.

You derive information exclusively from source data and cite everything using markdown footnotes. Your available tools are:

1. memory_manager: Store verified facts by specifying 'action': 'store', 'content', 'importance' (0.0-2.0), and optional 'reason'.

2. visit: Extract webpage content by providing 'url' and optional 'extract' prompt for focused extraction.

3. search/google_search: Query the web using 'query' parameter and optional 'num_results' (1-10, default 3).

When using these tools:
- First identify key questions and resolve ambiguity by asking for clarification
- Use search with focused queries to find authoritative sources
- Visit pages with targeted extract prompts to get specific data
- Cross-validate claims across multiple sources

When storing memories, capture verified information with citations. Rate importance on a 0.0-2.0 scale based on:
- Supporting evidence (is this fact directly cited from reliable sources?)
- Relevance to user's context (is this useful context for the future?)
- Citation quality (are sources authoritative and current?)

You MUST include markdown footnote citations for all factual claims. When you encounter any entity, assign them short, unique identifiers as subscripts$_{like_this}$. Include very short editorial notes in superscript markers$^{like_this}$.  You may include footnoted regarding self-thought or longer editorial notes. If sources conflict or information is ambiguous, immediately ask the user for clarification."""

    if relevant_memories:
        system_message += (
            "\n\nRelevant memories from previous conversations:\n" + relevant_memories
        )

    try:
        # Format messages for the API
        messages = []

        # Add historical context
        for msg in history:
            role = "assistant" if msg["role"] == "assistant" else "user"
            messages.append({"role": role, "content": msg["content"]})

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        # Get available search tools
        tools = get_search_tools() if include_web_search else []

        # Make request to Claude with tool calling enabled
        if progress_callback:
            progress_callback("Making request to Claude")

        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4000,
            system=system_message,
            messages=messages,
            tools=tools,
            temperature=0.5,
        )

        # Process any tool calls recursively
        while response.stop_reason == "tool_use":
            # Get the tool use block
            tool_use = next(
                block for block in response.content if block.type == "tool_use"
            )
            tool_name = tool_use.name
            tool_args = tool_use.input
            tool_counter.count_tool(tool_name)

            # Execute the tool
            if progress_callback:
                progress_callback(f"Executing tool: {tool_name}", tool_args)

            tool_result = execute_tool(tool_name, tool_args)
            if progress_callback:
                progress_callback("Tool execution complete", tool_result)

            # Convert tool result to string if it's a dict
            if isinstance(tool_result, dict):
                tool_result = json.dumps(tool_result)

            # Add tool result to messages and continue conversation
            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": tool_result,
                        }
                    ],
                }
            )

            # Make another request with the tool result
            if progress_callback:
                progress_callback("Continuing conversation with tool results")

            response = anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                system=system_message,
                messages=messages,
                tools=tools,
                temperature=0.5,
            )

        # Get the final text response
        final_response = next(
            (block.text for block in response.content if hasattr(block, "text")),
            "I apologize, but I was unable to generate a response. Please try again.",
        )

        # Log if no text block was found
        if not any(hasattr(block, "text") for block in response.content):
            print(
                f"Response did not include text and had stop_reason of {response.stop_reason}"
            )

        # Store recent questions information in memory
        # st.session_state.memory_manager.add_memory(
        #     f"User asked: {prompt}\nAssistant responded: {final_response}",
        #     importance=1.0,
        # )

        return final_response

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return f"Error: {str(e)}"


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
