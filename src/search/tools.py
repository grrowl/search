from typing import Dict, List
from .providers.duckduckgo import execute_duckduckgo_search
from .providers.serpapi import execute_serpapi_search
from .providers.firecrawl import execute_firecrawl

def get_search_tools() -> List[Dict]:
    """Get the available search tools based on configuration"""
    tools = []
    
    # Add DuckDuckGo search
    tools.append({
        "name": "search",
        "description": """Find information across the web. Returns titles, snippets and links. 
        Use to research real-time and up-to-date information from anywhere on the internet.""",
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
    })
    
    return tools

def execute_tool(tool_name: str, tool_args: dict) -> str:
    """Execute the requested search tool with the given input"""
    try:
        if tool_name == "search":
            query = tool_args.get("query")
            num_results = min(max(tool_args.get("num_results", 5), 1), 15)
            return execute_duckduckgo_search(query, num_results)
        elif tool_name == "google_search":
            query = tool_args.get("query")
            num_results = min(max(tool_args.get("num_results", 5), 1), 15)
            return execute_serpapi_search(query, num_results)
        elif tool_name == "visit":
            url = tool_args.get("url")
            prompt = tool_args.get("prompt")
            return execute_firecrawl(url, prompt)
        else:
            return {"error": f"Unknown tool '{tool_name}'", "is_error": True}
    except Exception as e:
        return {"error": f"Error executing {tool_name}: {str(e)}", "is_error": True}
