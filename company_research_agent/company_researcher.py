from typing import TypedDict, List, Annotated, Literal, Dict, Union, Optional 
from datetime import datetime
from langchain_core.pydantic_v1 import BaseModel, Field
from tavily import AsyncTavilyClient
import json
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, add_messages
import os
from dotenv import load_dotenv

load_dotenv()

# Define the research state
class ResearchState(TypedDict):
    report: str
    documents: Dict[str, Dict[Union[str, int], Union[str, float]]]
    messages: Annotated[list[AnyMessage], add_messages]
    research_count: int

# Define the structure for the model's response, which includes citations.
class Citation(BaseModel):
    source_id: str = Field(
        ...,
        description="The url of a SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )


class QuotedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""
    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources. Include any relevant sources in the answer as markdown hyperlinks. For example: 'This is a sample text ([url website](url))'"
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )
    
# Add Tavily's arguments to enhance the web search tool's capabilities
class TavilyQuery(BaseModel):
    query: str = Field(description="sub query")
    topic: str = Field(description="type of search, should be 'general' or 'news'. Choose 'news' ONLY when the company you searching is publicly traded and is likely to be featured on popular news")
    days: int = Field(description="number of days back to run 'news' search")
    raw_content: bool = Field(description="include raw content from found sources, use it ONLY if you need more information besides the summary content provided")
    domains: Optional[List[str]] = Field(default=None, description="list of domains to include in the research. Useful when trying to gather more detailed information.")
 

# Define the args_schema for the tavily_search tool using a multi-query approach, enabling more precise queries for Tavily.
class TavilySearchInput(BaseModel):
    sub_queries: List[TavilyQuery] = Field(description="set of sub-queries that can be answered in isolation")


@tool("tavily_search", args_schema=TavilySearchInput, return_direct=True)
async def tavily_search(sub_queries: List[TavilyQuery]):
    """Perform searches for each sub-query using the Tavily search tool concurrently."""  
    # Define a coroutine function to perform a single search with error handling
    async def perform_search(itm):
        try:
            # Add date to the query as we need the most recent results
            query_with_date = f"{itm.query} {datetime.now().strftime('%m-%Y')}"
            # Attempt to perform the search, hardcoding days to 7 (days will be used only when topic is news)
            response = await tavily_client.search(query=query_with_date, topic=itm.topic, days=itm.days, include_raw_content=itm.raw_content, max_results=5)
            return response['results']
        except Exception as e:
            # Handle any exceptions, log them, and return an empty list
            print(f"Error occurred during search for query '{itm.query}': {str(e)}")
            return []
    
    # Run all the search tasks in parallel
    search_tasks = [perform_search(itm) for itm in sub_queries]
    search_responses = await asyncio.gather(*search_tasks)
    
    # Combine the results from all the responses
    search_results = []
    for response in search_responses:
        search_results.extend(response)
    
    return search_results


tools = [tavily_search]
tools_by_name = {tool.name: tool for tool in tools}
tavily_client = AsyncTavilyClient()
model = ChatOpenAI(model="gpt-4o-mini",temperature=0).bind_tools(tools)

# Define an async custom tool node to store Tavily's search results for improved processing and filtering.
async def tool_node(state: ResearchState):
    docs = state.get('documents', {})
    docs_str = ""
    msgs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        new_docs = await tool.ainvoke(tool_call["args"])
        for doc in new_docs:
            if not docs or doc['url'] not in docs:
                docs[doc['url']] = doc
                docs_str += json.dumps(doc)
        msgs.append(ToolMessage(content=f"Found the following new documents: {docs_str}", tool_call_id=tool_call["id"]))
    return {
        "messages": msgs, 
        "documents": docs,
        "research_count": state.get('research_count', 0)
    }
    
# Invoke the model with research tools to gather information about the company.     
def call_model(state: ResearchState):
    prompt = f"""Today's date is {datetime.now().strftime('%d/%m/%Y')}.
    You are an expert researcher tasked with preparing a weekly report on recent developments.
    Your current objective is to gather detailed information about any significant events that occurred in the past week.\n
    """
    messages = state.get('messages', []) + [SystemMessage(content=prompt)]
    response = model.invoke(messages)
    return {"messages": [response], "research_count": state.get('research_count', 0)}
    

# Define the function that decides whether to continue research using tools or proceed to writing the report
def should_continue(state: ResearchState) -> Literal["tools", "research"]:
    messages = state.get('messages', [])
    last_message = messages[-1]
    
    # Incrementar el contador solo cuando procesamos herramientas
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        state['research_count'] = state.get('research_count', 0) + 1
        return "tools"
    
    # Si ya alcanzamos el límite de investigación, terminamos
    if state.get('research_count', 0) >= 3:
        return "research"
    
    # Si no hay tool_calls, continuamos con research
    return "research"

# Define the function to write the report based on the retrieved documents.
def write_report(state: ResearchState):
    prompt = f"""Today's date is {datetime.now().strftime('%d/%m/%Y')}\n.
    You are an expert researcher, writing a weekly report about recent events.\n
    Your task is to write an in-depth, well-written, and detailed report based on the provided documents.\n
    Here are all the documents you gathered so far:\n{state.get('documents', {})}\n
    Use only the relevant and most recent documents.
    Respond in Spanish.""" 
    messages = [state['messages'][-1]] + [SystemMessage(content=prompt)]
    response = model.with_structured_output(QuotedAnswer).invoke(messages)
    return {
        "messages": [AIMessage(content=f"Generated Report:\n{response.answer}")], 
        "report": response.answer,
        "research_count": state.get('research_count', 0)
    }

# Define a graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("research", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint
workflow.set_entry_point("research")

# Add conditional edges
workflow.add_conditional_edges(
    "research",
    should_continue,
    {
        "tools": "tools",
        "research": END
    }
)

# Add edge from tools to research
workflow.add_edge("tools", "research")
workflow.add_edge("research", END)

app = workflow.compile()