"""
Label Resolver for wbgapi360
Converts economy ISO codes to human-readable names while preserving codes for choropleth maps.
"""
import pandas as pd
from typing import Optional
from .iso_mapping import get_name, get_code, ISO_3166_NAMES


class LabelResolver:
    """
    Resolves economy codes to human-readable labels.
    Preserves original codes in REF_AREA_CODE column for map compatibility.
    """
    
    def __init__(self):
        self._mapping = ISO_3166_NAMES
    
    def get_economy_name(self, code: str) -> str:
        """
        Get human-readable name for economy code.
        
        Args:
            code: ISO 3166-1 Alpha-3 code (e.g., "USA")
            
        Returns:
            Human-readable name (e.g., "United States")
            Falls back to original code if not found.
        """
        return get_name(code)
    
    def get_economy_code(self, name: str) -> str:
        """
        Get ISO code for human-readable name (reverse lookup).
        
        Args:
            name: Country name (e.g., "United States")
            
        Returns:
            ISO code (e.g., "USA")
            Falls back to original name if not found.
        """
        return get_code(name)
    
    def resolve_dataframe(
        self, 
        df: pd.DataFrame, 
        column: str = 'REF_AREA',
        preserve_codes: bool = True
    ) -> pd.DataFrame:
        """
        Replace economy codes with human-readable names in DataFrame.
        Optionally preserves original codes in a separate column for map compatibility.
        
        Args:
            df: DataFrame with economy codes
            column: Name of column containing codes (default: 'REF_AREA')
            preserve_codes: If True, adds REF_AREA_CODE column with original codes
            
        Returns:
            DataFrame with resolved names (and preserved codes if requested)
        """
        if column not in df.columns and column not in df.index.names:
            # Column doesn't exist, return as-is
            return df
        
        df_copy = df.copy()
        
        # Handle both column and index scenarios
        if column in df_copy.columns:
            # Column case
            if preserve_codes:
                # Preserve original codes before transformation
                df_copy[f'{column}_CODE'] = df_copy[column]
            
            # Apply name resolution
            df_copy[column] = df_copy[column].apply(self.get_economy_name)
            
        elif column in df_copy.index.names:
            # Index case (MultiIndex or single index)
            if preserve_codes:
                # Extract original codes from index
                if isinstance(df_copy.index, pd.MultiIndex):
                    # MultiIndex: Find level with REF_AREA
                    level_idx = df_copy.index.names.index(column)
                    codes = df_copy.index.get_level_values(level_idx)
                    df_copy[f'{column}_CODE'] = codes
                else:
                    # Single index
                    df_copy[f'{column}_CODE'] = df_copy.index
            
            # Apply name resolution to index
            if isinstance(df_copy.index, pd.MultiIndex):
                # MultiIndex: Update specific level
                level_idx = df_copy.index.names.index(column)
                new_index = df_copy.index.set_levels(
                    df_copy.index.levels[level_idx].map(self.get_economy_name),
                    level=level_idx
                )
                df_copy.index = new_index
            else:
                # Single index
                df_copy.index = df_copy.index.map(self.get_economy_name)
        
        return df_copy
    
    def has_codes(self, df: pd.DataFrame, column: str = 'REF_AREA') -> bool:
        """
        Check if DataFrame contains ISO codes (vs human names).
        
        Args:
            df: DataFrame to check
            column: Column name to inspect
            
        Returns:
            True if column contains 3-letter codes, False if names
        """
        if column not in df.columns and column not in df.index.names:
            return False
        
        # Get sample values
        if column in df.columns:
            sample = df[column].dropna().head(10)
        else:
            if isinstance(df.index, pd.MultiIndex):
                level_idx = df.index.names.index(column)
                sample = pd.Series(df.index.get_level_values(level_idx)[:10])
            else:
                sample = pd.Series(df.index[:10])
        
        # Heuristic: ISO codes are 3 uppercase letters
        if len(sample) == 0:
            return False
        
        first_val = str(sample.iloc[0])
        return len(first_val) == 3 and first_val.isupper() and first_val.isalpha()
