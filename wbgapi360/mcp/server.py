from typing import List, Dict, Any, Optional, Union
import asyncio
import pandas as pd
import tempfile
import os
import datetime
import logging
import json
from io import StringIO
from httpx import HTTPStatusError
from fastmcp import FastMCP

# Internal imports
from wbgapi360.core.client import Data360Client
from wbgapi360.data.builder import DataBuilder
from wbgapi360.search.engine import SearchEngine
from wbgapi360.visual.charts import Visualizer

# Setup logger
logger = logging.getLogger("wbgapi360.server")

# Component initialization
mcp = FastMCP("wbgapi360-agent")
# Lazy Singletons
_client_instance: Optional[Data360Client] = None
_search_engine_instance: Optional[SearchEngine] = None

viz = Visualizer()  # Single instance for visualization is safe as it's synchronous

def _get_client() -> Data360Client:
    """Lazy initialize the Data360Client."""
    global _client_instance
    if _client_instance is None:
        _client_instance = Data360Client()
    return _client_instance

def _get_search_engine() -> SearchEngine:
    """Lazy initialize the SearchEngine."""
    global _search_engine_instance
    if _search_engine_instance is None:
        client = _get_client()
        _search_engine_instance = SearchEngine(client)
    return _search_engine_instance

# --- RAW IMPLEMENTATIONS (Core Functions) ---

async def _search_indicators(query: str, limit: int = 10, database_id: str = "WB_WDI") -> List[Dict[str, Any]]:
    """Search for World Bank indicators using Smart Ranking."""
    engine = _get_search_engine()
    raw_results = await engine.semantic_explore(query, database_id=database_id)
    
    formatted = [
        {
            "code": item.get('series_description', {}).get('idno'),
            "name": item.get('series_description', {}).get('name'),
            "database_id": item.get('series_description', {}).get('database_id'),
            "definition": (item.get('series_description', {}).get('definition', '') or '')[:200] + "..."
        }
        for item in raw_results[:limit]
    ]
        
    return formatted

async def _get_data(
    indicator_code: Union[str, List[str]], 
    economies: List[str], 
    years: Union[int, List[int]] = 5,
    database_id: str = "WB_WDI",
    labels: bool = False,  # NEW: Enable human-readable names
    as_frame: bool = False
) -> Union[str, pd.DataFrame]:
    """
    Fetch data with auto-correction logic (Restored from v0.1).
    """
    # Instantiate Builder per request (Thread-Safety)
    client = _get_client()
    builder = DataBuilder(client, dataset_id=database_id)
    
    # --- PROTOCOLO DE CORRECCIÓN DE IDs (Normalization) ---
    raw_codes = indicator_code if isinstance(indicator_code, list) else [indicator_code]
    codes = []
    
    if database_id == "WB_WDI":
        for c in raw_codes:
            clean_code = c.replace('.', '_')
            if not clean_code.startswith('WB_WDI_'):
                clean_code = f"WB_WDI_{clean_code}"
            codes.append(clean_code)
    else:
        codes = raw_codes

    b = builder.indicator(codes).economy(economies)
    
    # --- LÓGICA DE VENTANA DE TIEMPO (Range Enforcement) ---
    mrv_mode = False
    mrv_target_col = None
    
    if isinstance(years, int):
        current_year = datetime.datetime.now().year
        
        if years == 1:
            # MRV (Most Recent Value) Mode
            mrv_mode = True
            mrv_target_col = str(current_year) 
            start_year = current_year - 5
            b.time(f"{start_year}:{current_year}")
        else:
            # Ventana estándar (Trends)
            start_year = current_year - years
            b.time(f"{start_year}:{current_year}")
            
    elif isinstance(years, list):
        if years:
            min_y, max_y = min(years), max(years)
            b.time(f"{min_y}:{max_y}")
        else:
            current_year = datetime.datetime.now().year
            b.time(f"{current_year-5}:{current_year}")

    try:
        # Pivot=True es clave para scatter plots y para tener años como columnas
        df = await b.to_dataframe(pivot=True)
        
        if df.empty:
            return pd.DataFrame() if as_frame else "No data found for the specified parameters."
            
        # --- FILTRADO MRV (Post-Processing) ---
        if mrv_mode:
            year_cols = [c for c in df.columns if str(c).isdigit()]
            if year_cols:
                year_cols.sort()
                mrv_values = df[year_cols].ffill(axis=1).iloc[:, -1]
                df = df.drop(columns=year_cols)
                df[mrv_target_col] = mrv_values
                df = df.dropna(subset=[mrv_target_col])
        
        if as_frame:
            return df.reset_index()

        return df.reset_index().to_json(orient='records')

    except HTTPStatusError as e:
        return f"API Error {e.response.status_code}: {str(e)}"
    except Exception as e:
        return f"System Error: {str(e)}"

async def _compare_countries(
    economies: List[str],
    indicators: Union[str, List[str]],
    years: int = 10,
    normalize: bool = False
) -> str:
    """
    Compare multiple countries across one or more indicators.
    
    Args:
        economies: List of country codes
        indicators: Single indicator or list of indicators
        years: Years of historical data
        normalize: If True, normalize values to 0-100 scale for comparison
    
    Returns:
        JSON with combined data suitable for multi-line charts
    """
    try:
        # Handle single indicator
        if isinstance(indicators, str):
            indicators = [indicators]
        
        all_data = []
        
        # Fetch data for each indicator
        for indicator in indicators:
            data = await _get_data(indicator, economies, years, as_frame=True)
            
            if isinstance(data, str):  # Error case
                continue
                
            if not data.empty:
                # Add indicator name to differentiate
                data['INDICATOR'] = indicator
                all_data.append(data)
        
        if not all_data:
            return "No data found for specified indicators and countries"
        
        # Combine all dataframes
        combined = pd.concat(all_data, ignore_index=True)
        
        # Normalize if requested
        if normalize and not combined.empty:
            value_col = combined.columns[0]  # Assuming first col is the value
            # Normalize to 0-100 scale
            min_val = combined[value_col].min()
            max_val = combined[value_col].max()
            if max_val > min_val:
                combined[value_col] = ((combined[value_col] - min_val) / (max_val - min_val)) * 100
        
        return combined.to_json(orient='records')
        
    except Exception as e:
        logger.error(f"Failed to compare countries", exc_info=True)
        return f"Comparison Error: {str(e)}"

async def _analyze_trend(
    indicator: str,
    economy: str,
    years: int = 20,
    include_stats: bool = True
) -> str:
    """
    Analyze time-series trend with statistical insights.
    
    Args:
        indicator: Indicator code
        economy: Country code
        years: Years of historical data
        include_stats: Include statistical analysis
    
    Returns:
        JSON with data and statistical insights
    """
    try:
        # Fetch data
        data = await _get_data(indicator, [economy], years, as_frame=True)
        
        if isinstance(data, str) or data.empty:
            return "No data available for analysis"
        
        # Reset index to get TIME_PERIOD as column
        df = data.reset_index()
        
        # Get value column (first non-index column)
        value_col = [c for c in df.columns if c not in ['TIME_PERIOD', 'REF_AREA']][0]
        
        # Basic stats
        result = {
            "data": df.to_dict('records'),
            "meta": {
                "indicator": indicator,
                "economy": economy,
                "years_analyzed": len(df)
            }
        }
        
        if include_stats and len(df) > 1:
            values = df[value_col].dropna()
            
            if len(values) >= 2:
                # Calculate growth rates
                first_val = values.iloc[0]
                last_val = values.iloc[-1]
                
                if first_val != 0:
                    total_growth = ((last_val - first_val) / abs(first_val)) * 100
                    avg_annual_growth = total_growth / len(values)
                else:
                    total_growth = 0
                    avg_annual_growth = 0
                
                # Calculate volatility (std dev)
                volatility = values.std()
                
                # Trend direction
                if avg_annual_growth > 1:
                    trend = "increasing"
                elif avg_annual_growth < -1:
                    trend = "decreasing"
                else:
                    trend = "stable"
                
                result["statistics"] = {
                    "average_value": float(values.mean()),
                    "min_value": float(values.min()),
                    "max_value": float(values.max()),
                    "total_growth_pct": round(total_growth, 2),
                    "avg_annual_growth_pct": round(avg_annual_growth, 2),
                    "volatility": round(volatility, 2),
                    "trend_direction": trend,
                    "latest_value": float(last_val)
                }
        
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Failed to analyze trend", exc_info=True)
        return f"Analysis Error: {str(e)}"

async def _rank_countries(
    indicator: str,
    region: Optional[str] = None,
    year: Optional[int] = None,
    top_n: int = 20
) -> str:
    """
    Rank countries by indicator value.
    
    Args:
        indicator: Indicator code
        region: Optional region filter (latam, africa, asia, europe, mena)
        year: Specific year (None = most recent)
        top_n: Number of top countries to return
    
    Returns:
        JSON with ranked countries
    """
    try:
        # Define region mappings
        REGION_COUNTRIES = {
            'latam': ['ARG', 'BOL', 'BRA', 'CHL', 'COL', 'CRI', 'ECU', 'SLV', 'GTM', 
                     'HND', 'MEX', 'NIC', 'PAN', 'PRY', 'PER', 'URY', 'VEN'],
            'africa': ['DZA', 'AGO', 'BEN', 'BWA', 'BFA', 'CMR', 'COD', 'EGY', 'ETH',
                      'GHA', 'KEN', 'MAR', 'NGA', 'SEN', 'ZAF', 'TZA', 'UGA', 'ZMB'],
            'asia': ['BGD', 'CHN', 'IND', 'IDN', 'JPN', 'KOR', 'MYS', 'PAK', 'PHL',
                    'SGP', 'THA', 'VNM'],
            'europe': ['AUT', 'BEL', 'CZE', 'DNK', 'FIN', 'FRA', 'DEU', 'GRC', 'HUN',
                      'IRL', 'ITA', 'NLD', 'NOR', 'POL', 'PRT', 'ESP', 'SWE', 'CHE', 'GBR'],
            'mena': ['DZA', 'EGY', 'IRN', 'IRQ', 'ISR', 'JOR', 'KWT', 'LBN', 'MAR',
                    'OMN', 'QAT', 'SAU', 'TUN', 'ARE']
        }
        
        # Determine economies to query
        if region and region.lower() in REGION_COUNTRIES:
            economies = REGION_COUNTRIES[region.lower()]
        else:
            # Top 50 economies globally
            economies = ['USA', 'CHN', 'JPN', 'DEU', 'GBR', 'FRA', 'IND', 'ITA', 'BRA', 'CAN',
                        'KOR', 'RUS', 'AUS', 'ESP', 'MEX', 'IDN', 'NLD', 'SAU', 'TUR', 'CHE',
                        'POL', 'ARG', 'BEL', 'THA', 'AUT', 'NGA', 'IRN', 'ARE', 'NOR', 'ISR',
                        'IRL', 'SGP', 'ZAF', 'HKG', 'DNK', 'PHL', 'MYS', 'COL', 'PAK', 'CHL',
                        'FIN', 'VNM', 'BGD', 'EGY', 'PER', 'CZE', 'PRT', 'GRC', 'NZL', 'ROM']
        
        # Fetch data (1 year for ranking)
        years_to_fetch = 1 if year is None else 1
        data = await _get_data(indicator, economies, years=years_to_fetch, as_frame=True, labels=True)
        
        if isinstance(data, str) or data.empty:
            return "No data available for ranking"
        
        df = data.reset_index()
        
        # Get value column
        value_col = [c for c in df.columns if c not in ['TIME_PERIOD', 'REF_AREA']][0]
        
        # Sort by value (descending)
        df_sorted = df.sort_values(value_col, ascending=False).head(top_n)
        
        # Add rank
        df_sorted['rank'] = range(1, len(df_sorted) + 1)
        
        # Select relevant columns
        result = df_sorted[['rank', 'REF_AREA', value_col]].to_dict('records')
        
        return json.dumps({
            "indicator": indicator,
            "region": region or "global",
            "year": year or "latest",
            "rankings": result
        })
        
    except Exception as e:
        logger.error(f"Failed to rank countries", exc_info=True)
        return f"Ranking Error: {str(e)}"

def _plot_chart(chart_type: str, data: Union[str, pd.DataFrame], title: str = "Chart", subtitle: str = "", **kwargs) -> str:
    """
    Generates a chart image file from JSON data using a Dispatcher Pattern.
    Supports advanced parameters: region, bbox, bins, labels, etc.
    """
    # Guard clause for error strings
    if isinstance(data, str) and ("No data" in data or "Error" in data[:20]):
        return "Cannot generate chart. The data retrieval failed."
    
    # Guard clause for empty DataFrame
    if isinstance(data, pd.DataFrame) and data.empty:
         return "Cannot generate chart. Data is empty."

    try:
        # 1. Parse Data (if needing parsing)
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.read_json(StringIO(data), orient='records')
        
        # 2. Prepare Path
        fd, path = tempfile.mkstemp(suffix=f"_{chart_type}.png", prefix="wbg_plot_")
        os.close(fd)

        # 3. Dispatcher Logic (Clean Architecture)
        dispatch_table = {
            'trend': viz.plot_trend,
            'line': viz.plot_trend,
            'bar': viz.plot_bar,
            'column': viz.plot_column,
            'scatter': viz.plot_scatter,
            'map': viz.plot_map,
            'map_bubble': viz.plot_map_bubble,
            'map_diverging': viz.plot_map_diverging,
            'map_categorical': viz.plot_map_categorical,
            'dumbbell': viz.plot_dumbbell,
            'stacked': viz.plot_stacked_bar,
            'stacked_bar': viz.plot_stacked_bar,
            'area': viz.plot_area,
            'heatmap': viz.plot_heatmap,
            'bump': viz.plot_bump,
            'treemap': viz.plot_treemap,
            'donut': viz.plot_donut,
            'pie': viz.plot_donut
        }

        plot_func = dispatch_table.get(chart_type)

        if not plot_func:
            return f"Error: Chart type '{chart_type}' not supported. Available: {list(dispatch_table.keys())}"

        # 4. Execute with kwargs support
        plot_func(df, title=title, subtitle=subtitle, save_path=path, **kwargs)
            
        return f"Image saved to: {path}"

    except Exception as e:
        # --- FIX: OBSERVABILITY (P0 BLOCKER) ---
        logger.error(f"Failed to generate {chart_type} chart", exc_info=True)
        return f"Visualization Error: {str(e)}"

# --- MCP TOOLS (Decorated) ---

@mcp.tool()
async def search_indicators(query: str, limit: int = 10, database_id: str = "WB_WDI") -> List[Dict[str, Any]]:
    """Search for World Bank indicators using Smart Ranking."""
    return await _search_indicators(query, limit, database_id)

@mcp.tool()
async def get_data(
    indicator_code: Union[str, List[str]], 
    economies: List[str], 
    years: int = 5,
    labels: bool = False
) -> str:
    """Fetch World Bank data with auto-correction. Set labels=True for human-readable country names."""
    return await _get_data(indicator_code, economies, years, labels=labels)

@mcp.tool()
def plot_chart(
    chart_type: str, 
    data_json: str, 
    title: str = "Chart", 
    subtitle: str = "",
    region: Optional[str] = None,
    bbox: Optional[List[float]] = None,
    bins: Optional[List[float]] = None,
    category_labels: Optional[List[str]] = None
) -> str:
    """
    Generate chart with FT style. 
    Advanced: region (latam/africa/europe), bbox, bins, category_labels for map_categorical.
    """
    kwargs = {}
    if region:
        kwargs['region'] = region
    if bbox:
        kwargs['bbox'] = bbox
    if bins:
        kwargs['bins'] = bins
    if category_labels:
        kwargs['labels'] = category_labels
    
    return _plot_chart(chart_type, data_json, title, subtitle, **kwargs)

# --- ADVANCED ANALYTICS TOOLS ---

@mcp.tool()
async def compare_countries(
    economies: List[str],
    indicators: Union[str, List[str]],
    years: int = 10,
    normalize: bool = False
) -> str:
    """
    Compare multiple countries across indicators.
    
    Examples:
        - compare_countries(["PER", "CHL"], "NY.GDP.PCAP.CD", years=10)
        - compare_countries(["USA", "CHN"], ["FP.CPI.TOTL", "NY.GDP.MKTP.KD.ZG"], normalize=True)
    
    Args:
        economies: List of ISO3 country codes
        indicators: One or more indicator codes
        years: Years of historical data (default: 10)
        normalize: Normalize to 0-100 scale for comparison (default: False)
    
    Returns:
        JSON with combined data for multi-line comparison chart
    """
    return await _compare_countries(economies, indicators, years, normalize)

@mcp.tool()
async def analyze_trend(
    indicator: str,
    economy: str,
    years: int = 20,
    include_stats: bool = True
) -> str:
    """
    Analyze time-series trend with statistical insights.
    
    Returns growth rates, volatility, trend direction, and more.
    
    Example:
        - analyze_trend("NY.GDP.MKTP.KD.ZG", "PER", years=15)
    
    Args:
        indicator: Indicator code
        economy: ISO3 country code
        years: Years of historical data (default: 20)
        include_stats: Include statistical analysis (default: True)
    
    Returns:
        JSON with data and statistics (avg growth, volatility, trend direction)
    """
    return await _analyze_trend(indicator, economy, years, include_stats)

@mcp.tool()
async def rank_countries(
    indicator: str,
    region: Optional[str] = None,
    top_n: int = 20
) -> str:
    """
    Rank countries by indicator value.
    
    Examples:
        - rank_countries("NY.GDP.PCAP.CD")  # Global GDP per capita ranking
        - rank_countries("SP.POP.TOTL", region="latam", top_n=10)  # Top 10 LATAM by population
    
    Args:
        indicator: Indicator code
        region: Region filter (latam, africa, asia, europe, mena) or None for global
        top_n: Number of top countries to return (default: 20)
    
    Returns:
        JSON with ranked countries and their values
    """
    return await _rank_countries(indicator, region, None, top_n)

# ==========================================
# CAPA 3: SYSTEM PROMPTS (The Brain)
# ==========================================

@mcp.prompt()
def world_bank_analyst() -> str:
    """
    Returns the System Prompt with the new Metadata Verification Strategy.
    """
    return """
ROLE: Senior Data Economist at the World Bank.

MISSION:
You assist users in retrieving, analyzing, and visualizing economic data using the `wbgapi360` tools.
Your goal is accuracy, academic rigor, and data-driven storytelling.

PROTOCOL:
1.  **SEARCH & VERIFY (CRITICAL STEP):**
    - Unless the user provides a specific code, use `search_indicators`.
    - **DO NOT blindly pick the first result.** The search engine is semantic but fallible.
    - **STRATEGY:** Look at the top 3 candidates. **READ the 'definition' field** in the metadata.
    - **CHECK:** Does the definition match the economic concept? (e.g., distinguish between "Net Migration Inflows" vs "Foreign Direct Investment Inflows").
    - If unsure between two codes, prefer the one from database "WB_WDI".

2.  **LANGUAGE & INPUT HANDLING:**
    - Recognize and preserve Latin characters (tildes 'á', 'ñ') in queries.
    - Process Spanish queries natively.

3.  **FETCH ROBUSTLY:**
    - Use `get_data` with the verified code.
    - Default to 10 years for trends, 1 year for rankings.

4.  **VISUALIZE STRATEGICALLY:**
    - 'Evolution' -> 'trend' chart.
    - 'Ranking' -> 'bar' chart.
    - 'Correlation' -> 'scatter' chart.
    - 'Geography' -> 'map' chart.

5.  **ERROR HANDLING:**
    - If data is empty, state: "Data not available for this specific query."
    - If search is ambiguous, ask the user for clarification.

TONE:
Professional, concise, and objective. Use Markdown for tables.
"""