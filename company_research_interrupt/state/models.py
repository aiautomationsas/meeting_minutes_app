from pydantic import BaseModel, Field
from typing import List

class TavilyQuery(BaseModel):
    query: str = Field(description="sub query")
    topic: str = Field(description="type of search")
    days: int = Field(description="number of days back")
    raw_content: bool = Field(description="include raw content")

class TavilySearchInput(BaseModel):
    sub_queries: List[TavilyQuery] = Field(description="set of sub-queries")