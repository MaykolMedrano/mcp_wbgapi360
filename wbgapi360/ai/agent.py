from ..core.client import Data360Client
from ..search.engine import SearchEngine
from ..data.builder import DataBuilder
from typing import Dict, Any, List
import logging

logger = logging.getLogger("wbgapi360")

class DataAgent:
    """
    The 'Smart' interface. Relies on the API's vector search 
    to resolve natural language to IDs.
    """
    def __init__(self, client: Data360Client):
        self.client = client
        self.search = SearchEngine(client)

    async def get_context(self, natural_query: str) -> Dict[str, Any]:
        """
        Understands the query using vector search and returns a DataContext.
        """
        # 1. Search for the indicator using semantic search
        # 1. Search for the indicator using semantic search, preferring WDI
        logger.info(f"[AI] Thinking about '{natural_query}'...")
        results = await self.search.semantic_explore(natural_query, database_id="WB_WDI")
        
        if not results:
            logger.info(f"[AI] No results found for '{natural_query}'.")
            return {"error": f"I couldn't find any relevant data for '{natural_query}' in the World Bank 360 API."}

        # 2. Pick the top result, but verify it has minimal checks
        # In a real agent, we might present the top 3 to the user if confidence is low.
        best_match = results[0]
        series_desc = best_match.get('series_description', {})
        indicator_id = series_desc.get('idno')
        name = series_desc.get('name')
        database_id = series_desc.get('database_id')
        
        if not indicator_id:
             return {"error": "Found a match but it lacked a valid Indicator ID."}

        logger.info(f"[AI] I found: {name} (ID: {indicator_id}, DB: {database_id})")
        
        return {
            "indicator": indicator_id,
            "database_id": database_id or "WB_WDI",
            "name": name,
            "raw_match": best_match
        }

    async def get_available_dimensions(self, indicator_id: str) -> Dict[str, List[str]]:
        """
        Queries /disaggregation to see what dims are valid.
        Returns a dict of dim_name -> list of valid codes.
        """
        try:
            # The disaggregation endpoint returns metadata about valid filters
            # We use the generic 'get_data' since disaggregation is a GET endpoint
            response = await self.client.get_data("/disaggregation", params={"indicatorId": indicator_id})
            
            # Response handling logic (simplified for prototype)
            # Assuming response structure is list of objects with dimension info
            dims = {}
            if isinstance(response, dict) and "value" in response:
                vals = response["value"]
                # Heuristic parsing of dimension metadata
                # Assuming structure might be list of dicts with 'id', 'name', or 'code'
                if isinstance(vals, list):
                    for v in vals:
                        # Try to find the dimension name and its valid codes
                        # This is speculative without the specific API contract for /disaggregation
                        # But we look for common keys.
                        dim_id = v.get('id') or v.get('code')
                        if dim_id:
                            # If the API returns valid values for this dimension, store them
                            # For now, we just map the dimension ID to a placeholder or count
                            dims[dim_id] = [] 
                            # If there's a nested 'values' list, capture it
                            if 'values' in v and isinstance(v['values'], list):
                                dims[dim_id] = [sub.get('id') for sub in v['values'] if 'id' in sub]
            
            return dims
        except Exception as e:
            logger.warning(f"[AI] Warning: Could not introspect dimensions: {e}")
            return {}

    async def ask(self, natural_query: str, economy: str = "WLD", years: int = 5):
        """
        End-to-end flow: Question -> Data.
        """
        ctx = await self.get_context(natural_query)
        if "error" in ctx:
            return ctx
            
        indicator_id = ctx["indicator"]
        database_id = ctx.get("database_id", "WB_WDI")
        
        # 3. Introspect (Smart Step)
        logger.info(f"[AI] Inspecting dimensions for {indicator_id}...")
        # For this prototype we just log that we are doing it. 
        # In a full version, we would check if 'economy' or 'years' is valid, 
        # or if we need to add specific filters based on the query text (e.g. 'rural').
        
        # 4. Fetch data via Builder
        logger.info(f"[AI] Fetching data for {economy} from {database_id}...")
        builder = DataBuilder(self.client, dataset_id=database_id)
        data = await builder.indicator(indicator_id).economy(economy).limit(years).get()
        
        return {
            "answer": f"Here is the data for '{ctx['name']}'",
            "data": data,
            "source_indicator": indicator_id,
            "name": ctx['name']
        }
