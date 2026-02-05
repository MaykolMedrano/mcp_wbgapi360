from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class Data360Response(BaseModel):
    """Generic response wrapper for Data360 API."""
    count: Optional[int] = None
    value: Union[List[Dict[str, Any]], Dict[str, Any], Any] = None

class SearchQuery(BaseModel):
    """Schema for Search V2 payload."""
    count: bool = True
    top: int = 10
    skip: int = 0
    search: str
    select: Optional[str] = None
    filter: Optional[str] = None
    orderby: Optional[str] = None
    # Vector specific
    vectorQueries: Optional[List[Dict[str, Any]]] = None

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: Optional[str] = None
