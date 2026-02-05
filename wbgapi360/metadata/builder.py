from typing import List, Dict, Any, Union, Optional
from ..core.client import Data360Client

class MetadataBuilder:
    def __init__(self, client: Data360Client):
        self.client = client
        self._filters: List[str] = []
        self._selects: List[str] = []
        self._top: Optional[int] = None
        self._skip: Optional[int] = None
        self._orderby: Optional[str] = None

    def where(self, field: str, operator: str, value: Any) -> 'MetadataBuilder':
        """
        Adds a filter condition.
        Usage: .where('series_description/database_id', 'eq', 'WB_WDI')
        """
        # Handle string quoting
        if isinstance(value, str):
            val_str = f"'{value}'"
        else:
            val_str = str(value)
        
        # Construct OData filter clause
        self._filters.append(f"{field} {operator} {val_str}")
        return self

    def raw_filter(self, filter_expression: str) -> 'MetadataBuilder':
        """Adds a raw OData filter string directly."""
        self._filters.append(filter_expression)
        return self

    def select(self, *fields: str) -> 'MetadataBuilder':
        """Specifies which fields to return."""
        self._selects.extend(fields)
        return self

    def top(self, n: int) -> 'MetadataBuilder':
        self._top = n
        return self
    
    def skip(self, n: int) -> 'MetadataBuilder':
        self._skip = n
        return self

    def order_by(self, field: str, direction: str = "asc") -> 'MetadataBuilder':
        self._orderby = f"{field} {direction}"
        return self

    def _build_query_string(self) -> str:
        parts = []
        if self._filters:
            # Join all filters with 'and'
            combined_filters = " and ".join(self._filters)
            parts.append(f"$filter={combined_filters}")
        
        if self._selects:
            parts.append(f"$select={','.join(self._selects)}")
            
        if self._top is not None:
            parts.append(f"$top={self._top}")
            
        if self._skip is not None:
            parts.append(f"$skip={self._skip}")
            
        if self._orderby is not None:
            parts.append(f"$orderby={self._orderby}")

        # The API expects the query to start with '&' if it follows other params, 
        # but here it is the sole payload content. Usually OData starts with $ or &.
        # OAS example: "&$filter=..."
        return "&" + "&".join(parts)

    async def get(self) -> List[Dict[str, Any]]:
        """Executes the metadata query."""
        query_str = self._build_query_string()
        payload = {"query": query_str}
        
        # OAS says response is 200 Success. Assuming standard value wrapper or direct list.
        # We'll need to inspect the actual response structure in testing.
        response = await self.client.post_data("/metadata", payload)
        
        # Safely unwrap typical OData array
        if isinstance(response, dict) and "value" in response:
             return response["value"]
        return response

