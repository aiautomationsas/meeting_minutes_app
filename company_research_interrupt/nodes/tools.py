from datetime import datetime
from langchain_core.tools import tool
from tavily import AsyncTavilyClient
import json

tavily_client = AsyncTavilyClient()

@tool("tavily_search")
async def tavily_search(query: str) -> str:
    """Search for information using Tavily."""
    try:
        response = await tavily_client.search(
            query=f"{query} {datetime.now().strftime('%m-%Y')}",
            search_depth="advanced",
            max_results=5
        )
        return json.dumps(response['results'], ensure_ascii=False)
    except Exception as e:
        print(f"Error en búsqueda: {str(e)}")
        return "[]"

tools = [tavily_search]
tools_by_name = {tool.name: tool for tool in tools}