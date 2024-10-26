import os
from serpapi import GoogleSearch

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

        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"SerpAPI search error: {str(e)}"
