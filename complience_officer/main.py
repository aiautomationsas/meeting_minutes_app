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
    user_query:str
    report: str
    documents: Dict[str, Dict[Union[str, int], Union[str, float]]]
    messages: Annotated[list[AnyMessage], add_messages]

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
    topic: str = Field(description="type of search, should be 'general' or 'news'")
    days: int = Field(description="number of days back to run 'news' search")
    raw_content: bool = Field(description="include raw content from found sources, use it ONLY if you need more information besides the summary content provided")
    include_domains: List[str] = Field(
        default=[
            "https://www.suin-juriscol.gov.co/",
            "https://www.riskglobalconsulting.com/boletin-informativo/guia-sobre-el-sagrilaft-en-colombia-implementacion-y-requisitos-para-su-empresa/",
            "https://www.supersociedades.gov.co/"
        ],
        description="list of domains to include in the research"
    )

# Define the args_schema for the tavily_search tool using a multi-query approach, enabling more precise queries for Tavily.
class TavilySearchInput(BaseModel):
    sub_queries: List[TavilyQuery] = Field(description="set of sub-queries that can be answered in isolation")

@tool("tavily_search", args_schema=TavilySearchInput, return_direct=True)
async def tavily_search(sub_queries: List[TavilyQuery]):
    """
    Realiza búsquedas web utilizando el servicio Tavily.

    Esta función toma una lista de consultas y realiza búsquedas web para cada una,
    utilizando los parámetros especificados en cada consulta. Los resultados de todas
    las búsquedas se combinan y se devuelven.

    Args:
        sub_queries (List[TavilyQuery]): Una lista de objetos TavilyQuery, cada uno
        especificando los parámetros para una búsqueda individual.

    Returns:
        List[Dict]: Una lista de resultados de búsqueda combinados de todas las consultas.
    """
    search_results = []
    for query in sub_queries:
        response = await tavily_client.search(
            query=query.query,
            topic=query.topic,
            days=query.days,
            include_raw_content=query.raw_content,
            max_results=5,
            include_domains=query.include_domains
        )
        search_results.extend(response['results'])
    
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
        if isinstance(new_docs, list):
            for doc in new_docs:
                if isinstance(doc, dict) and 'url' in doc:
                    # Make sure that this document was not retrieved before
                    if doc['url'] not in docs:
                        docs[doc['url']] = doc
                        docs_str += json.dumps(doc)
                else:
                    # If doc is not a dict or doesn't have 'url', we treat it as a string
                    docs_str += str(doc)
        else:
            # If new_docs is not a list, we treat it as a single response
            docs_str += str(new_docs)
        
        msgs.append(ToolMessage(content=f"Found the following new documents: {docs_str}", tool_call_id=tool_call["id"]))
    
    return {"messages": msgs, "documents": docs}
    
# Invoke the model with research tools to gather information about the company.     
def call_model(state: ResearchState):
    prompt = f"""
    Today's date is {datetime.now().strftime('%d/%m/%Y')}.\n
    You are an expert researcher, specialized in SARLAFT (Money Laundering and Terrorist Financing Risk Management System) and 
    SAGRILAFT (System of Self-Control and Integral Management of the Risk of Money Laundering, Terrorist Financing and Financing of the Proliferation of Weapons of Mass Destruction).

    Please search and collect:

    Current  laws and regulations related to this topic.

    Focus on the most recent and relevant information. Be sure to include the dates of the sources found.

    Its mission is to provide accurate and updated information on these compliance systems in response to the following user query: {state['user_query']}.\n

    Respond in Spanish.\n

    """
    messages = state['messages'] + [SystemMessage(content=prompt)]
    # print("state['messages']:",state['messages'])
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}
    

# Define the function that decides whether to continue research using tools or proceed to writing the report
def should_continue(state: ResearchState) -> Literal["tools", "write_report"]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user with citations)
    return "write_report"

# Define the function to write the report based on the retrieved documents.
def write_report(state: ResearchState):
    # Create the prompt
    prompt = f"""
    Today's date is {datetime.now().strftime('%d/%m/%Y')}\n.
    Your task is to provide accurate and up-to-date information on SARLAFT or SAGRILAFT compliance systems in response to user queries. 
    Use your expert knowledge to address the questions or concerns raised.\n

    When you receive an inquiry, follow these steps:

    1. Carefully analyze the user's query.
    Identify the key aspects related to SARLAFT or SAGRILAFT mentioned in the query.
    3. Provide relevant, accurate and up-to-date information on the issues identified.
    4. If the query requires specific information that you do not have, indicate that you do not have that exact information and suggest official sources where the user could find it.
    5. Make sure your response is clear, concise and easy to understand.

    When providing information, keep the following in mind:
    - Cite relevant laws, regulations or circulars where appropriate.
    - Explain technical concepts in a simple manner.
    - If relevant, mention regulatory entities such as the Superintendencia Financiera de Colombia or the UIAF (Unidad de Información y Análisis Financiero).\n

    User inquery:
    -----
    {state['user_query']}
    -----

    Here are all the documents you gathered so far:\n{state['documents']}\n
    Use only the relevant and most recent documents.

    Respond in Spanish language""" 
    messages = [state['messages'][-1]] + [SystemMessage(content=prompt)]
    response = model.with_structured_output(QuotedAnswer).invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [AIMessage(content=f"Generated Report:\n{response.answer}")], "report": response.answer}

# Define a graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("research", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("write_report", write_report)

# Set the entrypoint as route_query
workflow.set_entry_point("research")

# Determine which node is called next
workflow.add_conditional_edges(
    "research",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# Add a normal edge from `tools` to `route_query`.
# This means that after `tools` is called, `route_query` node is called next.
workflow.add_edge("tools", "research")
workflow.add_edge("write_report", END)

graph = workflow.compile()