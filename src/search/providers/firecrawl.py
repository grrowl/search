import os
from firecrawl import FirecrawlApp

def execute_firecrawl(url: str, prompt: str = None) -> str:
    """Execute a Firecrawl web extraction"""
    try:
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
        return f"Firecrawl error: {str(e)}"
