from typing import List, Dict, Any, Optional
from ..core.client import Data360Client
from ..core.models import SearchQuery

class SearchEngine:
    def __init__(self, client: Data360Client):
        self.client = client

    async def query(
        self, 
        term: str, 
        limit: int = 10, 
        skip: int = 0, 
        database_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Performs a search. If just text is provided, uses default search.
        Can be enhanced to use vectorQueries if the API supports explicit vector embeddings,
        or relies on the API's internal vector mapping for 'search' param.
        """
        # Based on OAS, /searchv2 accepts a complex body.
        # We start with simple keyword/semantic text search which the API handles intelligently.
        
        payload = {
            "search": term,
            "count": True,
            "top": limit,
            "skip": skip,
            "select": "series_description/idno, series_description/name, series_description/database_id"
        }

        if database_id:
            # OpenAPI Spec: filter expression to narrow down results.
            # Syntax: series_description/database_id eq 'WB_WDI'
            payload["filter"] = f"series_description/database_id eq '{database_id}'"

        return await self.client.post_data("/searchv2", payload)

    def _calculate_relevance(self, item_name: str, query: str) -> float:
        """
        Calculates semantic relevance score (Higher is better).
        Now with INTUITION (Fuzzy Logic).
        """
        import difflib
        
        name_lower = item_name.lower()
        query_lower = query.lower()
        
        # 0. Stopword filter (Basic noise reduction)
        stopwords = {'the', 'of', 'total', 'current', 'us$', 'percent', 'annual', 'index'}
        query_tokens = set(query_lower.split()) - stopwords
        name_tokens = set(name_lower.split()) - stopwords
        
        if not query_tokens: query_tokens = set(query_lower.split()) # Fallback
        
        score = 0.0
        
        # 1. Exact Token Overlap
        overlap = len(query_tokens.intersection(name_tokens))
        score += overlap * 10
        
        # 2. Phrase Match Bonus
        if query_lower in name_lower:
            score += 20
        elif query_lower.replace(" ", "") in name_lower.replace(" ", ""): # Spaceless match
             score += 15
             
        # 3. FUZZY MATCH (The Intuition)
        # SequenceMatcher gives a ratio 0.0-1.0
        fuzzy_ratio = difflib.SequenceMatcher(None, query_lower, name_lower).ratio()
        score += fuzzy_ratio * 30 # Weight fuzzy match heavily
        
        # 4. Length Penalty (Prefer concise matches)
        score -= len(name_lower) * 0.05
        
        # 5. Typos Handled:
        # If any token in query has high similarity to any token in name
        for qt in query_tokens:
             best_token_match = 0
             for nt in name_tokens:
                 ratio = difflib.SequenceMatcher(None, qt, nt).ratio()
                 if ratio > 0.8: # Threshold for "It's a typo"
                     best_token_match = max(best_token_match, ratio)
             score += best_token_match * 10
        
        return score

    def _smart_sort(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Sorts results using the improved Token-Based Relevance Algorithm.
        """
        clean_query = query.lower().strip()
        
        for item in results:
            # Safe extraction of name
            item_desc = item.get('series_description', {})
            name = item_desc.get('value', '') or item_desc.get('name', '')
            
            # Calculate Relevance
            score = self._calculate_relevance(name, clean_query)
            
            # Tie-breakers and metadata adjustments
            
            # Penalty for "noise" words common in disaggregations if query doesn't ask for them
            name_lower = name.lower()
            if "expenditure" in name_lower or "government" in name_lower:
                 # Only penalize if query didn't ask for them
                 if "expenditure" not in clean_query and "government" not in clean_query:
                    score -= 10 # Reduce score (Higher is better, so subtract)
                
            # Database Priority
            if item_desc.get('database_id') == 'WB_WDI':
                 score += 5 # Boost score
                
            # Store score
            item['_rank_score'] = score
            
        # Sort by score DESC (Higher is better)
        return sorted(results, key=lambda x: x.get('_rank_score', 0), reverse=True)

    async def semantic_explore(self, concept: str, database_id: str = "WB_WDI") -> List[Dict[str, Any]]:
        """
        Uses the search endpoint to find related indicators. 
        This is the 'Smart' discovery layer.
        
        Args:
            concept: Natural language query.
            database_id: Preferred database to filter by (default: WB_WDI for best global coverage).
        """
        # In a real SOTA implementation, we might tweak the 'vectorFilterModeSearch' parameter 
        # if the API documentation details it further. For now, we trust the v2 search endpoint.
        
        # We explicitly pass the database_id to filter at the source
        results = await self.query(concept, limit=50, database_id=database_id) # Createsed limit for re-ranking
        
        # Unwrap the values
        if results and isinstance(results, dict) and "value" in results:
             raw_list = results["value"]
             # Apply Smart Ranking
             ranked_list = self._smart_sort(raw_list, concept)
             return ranked_list
             
        return []
