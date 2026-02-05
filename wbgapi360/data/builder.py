from typing import List, Dict, Any, Optional, Union
from ..core.client import Data360Client
import logging

logger = logging.getLogger("wbgapi360")

class DataBuilder:
    def __init__(self, client: Data360Client, dataset_id: str = "WB_WDI"):
        self.client = client
        self.dataset_id = dataset_id
        self.params: Dict[str, Any] = {
            "DATABASE_ID": dataset_id,
            "top": 1000
        }

    def indicator(self, ids: Union[str, List[str]]) -> 'DataBuilder':
        """Set indicator(s)."""
        if isinstance(ids, list):
            self.params['INDICATOR'] = ",".join(ids)
        else:
            self.params['INDICATOR'] = ids
        return self

    def economy(self, codes: Union[str, List[str]]) -> 'DataBuilder':
        """Set economy/country filter (REF_AREA)."""
        if isinstance(codes, list):
            self.params['REF_AREA'] = ",".join(codes)
        else:
            self.params['REF_AREA'] = codes
        return self

    def time(self, periods: Union[str, List[str]]) -> 'DataBuilder':
        """Set time period(s). Supports lists or 'start:end' string."""
        if isinstance(periods, list):
            self.params['TIME_PERIOD'] = ",".join(map(str, periods))
        else:
            # Check for "start:end" syntax
            s_periods = str(periods)
            if ':' in s_periods:
                try:
                    parts = s_periods.split(':')
                    if len(parts) == 2:
                        # Use special range parameters discovered in OpenAPI spec
                        # capable of handling ranges without 417 errors
                        self.params['timePeriodFrom'] = parts[0]
                        self.params['timePeriodTo'] = parts[1]
                        
                        # Ensure we don't send conflicting TIME_PERIOD
                        if 'TIME_PERIOD' in self.params:
                            del self.params['TIME_PERIOD']
                    else:
                        # Fallback for weird formats
                        self.params['TIME_PERIOD'] = s_periods
                except Exception:
                    # Fallback
                     self.params['TIME_PERIOD'] = s_periods
            else:
                self.params['TIME_PERIOD'] = s_periods
        return self

    def limit(self, n: int) -> 'DataBuilder':
        self.params['top'] = n
        return self

    # --- Advanced Dimensions ---
    def sex(self, code: str) -> 'DataBuilder':
         self.params['SEX'] = code
         return self

    def urbanization(self, code: str) -> 'DataBuilder':
         self.params['URBANISATION'] = code
         return self

    async def get(self) -> List[Dict[str, Any]]:
        """Execute the query (single page)."""
        response = await self.client.get_data("/data", params=self.params)
        
        if isinstance(response, dict):
             # It might be in response['value']['value']
             val = response.get('value')
             if isinstance(val, dict) and 'value' in val:
                 return val['value']
             elif isinstance(val, list):
                 return val
        return []

    async def get_all(self, chunk_size=1000) -> List[Dict[str, Any]]:
        """
        Execute the query and automatically paginate to fetch ALL results.
        WARNING: This can be slow for very large queries.
        """
        all_data = []
        skip = 0
        self.params['top'] = chunk_size
        
        while True:
            self.params['skip'] = skip
            logger.debug(f"...fetching chunk starting at {skip}") # Simple progress logging
            page_data = await self.get()
            
            if not page_data:
                break
                
            all_data.extend(page_data)
            
            if len(page_data) < chunk_size:
                # Last page reached
                break
                
            skip += chunk_size
            
        return all_data

    async def to_dataframe(
        self, 
        fetch_all: bool = False, 
        pivot: bool = False,
        labels: bool = False  # Convert codes to human names
    ):
        """
        Execute the query and return a Pandas DataFrame.
        
        :param fetch_all: If True, auto-paginates to get ALL data.
        :param pivot: If True, reshapes data to have Years as columns (Wide format).
        :param labels: If True, converts economy codes (USA) to names (United States).
                      Preserves codes in REF_AREA_CODE column for map compatibility.
        """
        if fetch_all:
             data = await self.get_all()
        else:
             data = await self.get()

        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Pandas is required for to_dataframe(). Install it with `pip install pandas`.")
        
        if not data:
            logger.warning("[DataBuilder] Warning: Query returned no data. Returning empty DataFrame.")
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(data)
            
            if pivot:
                # Pivot: Index=[Economy, Indicator], Columns=[Year], Values=[Value]
                # Check required columns
                req_cols = ['REF_AREA', 'TIME_PERIOD', 'OBS_VALUE']
                if all(c in df.columns for c in req_cols):
                    # Ensure numeric values for pivot
                    df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
                    
                    df = df.pivot_table(
                        index=['REF_AREA', 'INDICATOR'], 
                        columns='TIME_PERIOD', 
                        values='OBS_VALUE'
                    )
                else:
                    logger.warning("[DataBuilder] Warning: Cannot pivot, missing standard columns.")
            
            else:
                # Heuristic to set a useful index if common columns exist
                candidates = ['REF_AREA', 'TIME_PERIOD', 'INDICATOR']
                index_cols = [c for c in candidates if c in df.columns]
                
                if index_cols:
                    df.set_index(index_cols, inplace=True)
            
            # Apply label resolution if requested
            if labels and not df.empty:
                from ..metadata.resolver import LabelResolver
                resolver = LabelResolver()
                df = resolver.resolve_dataframe(df, column='REF_AREA', preserve_codes=True)
                
            return df
        except Exception as e:
            logger.error(f"[DataBuilder] Error converting data to DataFrame: {e}")
            # Identify if it's a structure issue
            if data and isinstance(data, list):
                logger.debug(f"[DataBuilder] Sample record: {data[0]}")
            raise e

class DataInterface:
    def __init__(self, client: Data360Client):
        self.client = client
    
    def new_query(self, dataset_id="WB_WDI") -> DataBuilder:
        return DataBuilder(self.client, dataset_id)
        
    # Quick access helpers
    async def get(self, indicators, economies, time, dataset="WB_WDI"):
        return await self.new_query(dataset).indicator(indicators).economy(economies).time(time).get()
