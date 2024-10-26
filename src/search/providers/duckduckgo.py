from duckduckgo_search import DDGS

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
        return f"DuckDuckGo search error: {str(e)}"
