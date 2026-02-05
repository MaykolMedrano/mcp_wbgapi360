import re
from typing import List, Union

def normalize_indicator_code(code: str, database_id: str = "WB_WDI") -> str:
    """
    Normalizes indicator codes to match API requirements.
    e.g., 'NY.GDP.MKTP.KD' -> 'WB_WDI_NY_GDP_MKTP_KD'
    """
    clean_code = code.strip()
    
    if database_id == "WB_WDI":
        # Check if already prefixed
        if not clean_code.startswith("WB_WDI_"):
            # Replace dots with underscores
            clean_code = clean_code.replace('.', '_')
            clean_code = f"WB_WDI_{clean_code}"
            
    return clean_code

def normalize_codes(codes: Union[str, List[str]], database_id: str = "WB_WDI") -> List[str]:
    """Batch normalization."""
    if isinstance(codes, str):
        codes = [codes]
    
    return [normalize_indicator_code(c, database_id) for c in codes]

def resolve_economies(economies: Union[str, List[str]]) -> List[str]:
    """
    Placeholder for future name resolution logic.
    For now, just ensures list format.
    TODO: Implement smart name lookup (Brazil -> BRA).
    """
    if isinstance(economies, str):
        # Handle comma-separated string
        if "," in economies:
            return [e.strip() for e in economies.split(",")]
        return [economies]
    return economies
