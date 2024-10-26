from typing import List, Dict, Callable
from anthropic import Anthropic
import json
import os
import streamlit as st
from search import get_tools, execute_tool
from memory import ToolUsageCounter


def get_assistant_response(
    prompt: str,
    history: List[Dict],
    include_web_search: bool = True,
    progress_callback: Callable = None,
) -> str:
    """Get response from Claude API with memory and web search integration"""
    # Initialize tool counter
    tool_counter = ToolUsageCounter()

    # Get relevant memories and count them
    relevant_memories = st.session_state.memory_manager.get_relevant_memories(prompt)
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

Always include a `---` before continuing to footnotes. You MUST include markdown footnote citations for all factual claims. When you encounter any entity, assign them one-word unique identifiers as subscripts $_{like_this}$. Include very short editorial notes in superscript markers $^{like_this}$. Include editorial notes and any self-relfection on the response in additional bullet-points. If sources conflict or information is ambiguous, immediately ask the user for clarification."""

    if relevant_memories:
        system_message += (
            "\n\nRelevant memories from previous conversations:\n" + relevant_memories
        )

    try:
        # Get Anthropic API key
        api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("No API key found. Please set ANTHROPIC_API_KEY.")

        anthropic = Anthropic(api_key=api_key)

        # Format messages for the API
        messages = []

        # Add historical context
        for msg in history:
            role = "assistant" if msg["role"] == "assistant" else "user"
            messages.append({"role": role, "content": msg["content"]})

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        # Get available search tools
        tools = get_tools() if include_web_search else []

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

        return final_response

    except Exception as e:
        st.error(f"Error: {str(e)}")
        return f"Error: {str(e)}"
