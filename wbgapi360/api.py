import logging
import asyncio
import nest_asyncio
import pandas as pd
import datetime
import os
from typing import List, Dict, Any, Union, Optional

# --- CORE IMPORTS (Decoupled Architectue) ---
from wbgapi360.core.client import Data360Client
from wbgapi360.search.engine import SearchEngine
from wbgapi360.data.builder import DataBuilder
from wbgapi360.data.builder import DataBuilder
# from wbgapi360.visual.charts import Visualizer # LAZY LOADED
from wbgapi360.core.utils import normalize_codes, resolve_economies
from wbgapi360.core.utils import normalize_codes, resolve_economies
from wbgapi360.core.transformers import DataStandardizer
from wbgapi360.core.auditor import DataAuditor

# --- SETUP ---
logger = logging.getLogger("wbgapi360")
logger.addHandler(logging.NullHandler())

# Singletons for Sync API
# _client = Data360Client() -> PREVIOUSLY GLOBAL, NOW INSTANTIATED PER CALL TO FIX ASYNCIO LIFECYCLE
# Singletons for Sync API
# _client = Data360Client() -> PREVIOUSLY GLOBAL, NOW INSTANTIATED PER CALL TO FIX ASYNCIO LIFECYCLE
# _viz = Visualizer() -> NOW LAZY LOADED

def _run_sync(coro):
    """
    Helper privado: Ejecuta una corutina de forma síncrona.
    Aplica nest_asyncio solo si detecta un loop en ejecución (Jupyter).
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # Scoped Side-Effect: Correcto, solo bajo demanda.
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    else:
        return asyncio.run(coro)

# --- INTERNAL ASYNC LOGIC (Replicated from server to decouple) ---

# --- INTERNAL ASYNC LOGIC (Replicated from server to decouple) ---

async def _async_get_data(
    indicator: Union[str, List[str]], 
    economies: Union[str, List[str]], 
    years: int = 5,
    database_id: str = "WB_WDI",
    labels: bool = False,  # Convert codes to human names
    as_frame: bool = True  # Default to True for Python API
) -> Union[pd.DataFrame, str]:
    
    async with Data360Client() as client:
        # 1. Normalize Inputs
        codes = normalize_codes(indicator, database_id)
        eco_list = resolve_economies(economies)
        
        # 2. Builder
        builder = DataBuilder(client, dataset_id=database_id)
        builder.indicator(codes).economy(eco_list)
        
        # 3. Time Logic (MRV vs Trend)
        current_year = datetime.datetime.now().year
        
        if years == 1:
            # MRV Logic (Simplified)
            start_year = current_year - 5
            builder.time(f"{start_year}:{current_year}")
            # Note: True MRV filter logic is complex, for v0.3 we take simple range
            # and let the DataFrame user filter the last column if needed, 
            # or we could implement the full logic here. For now, strict range.
            mrv_mode = True
        else:
            mrv_mode = False
            start_year = current_year - years
            builder.time(f"{start_year}:{current_year}")

        # 4. Fetch
        try:
            # Pivot=True is standard for human analysis
            # labels=True converts codes to names (preserves codes for maps)
            df = await builder.to_dataframe(pivot=True, labels=labels)
            
            if df.empty:
                return pd.DataFrame() if as_frame else "No data found."
                
            # MRV Post-processing (Keep only valid latest value)
            if mrv_mode and not df.empty:
                def _is_year(c):
                    s = str(c)
                    if s.isdigit(): return True
                    try:
                        f = float(s)
                        return f.is_integer() and 1900 < f < 2100
                    except ValueError:
                        return False

                year_cols = [c for c in df.columns if _is_year(c)]
                if year_cols:
                    year_cols.sort()
                    # ffill to get last valid observation
                    last_vals = df[year_cols].ffill(axis=1).iloc[:, -1]
                    # Clean up structure
                    df = df.drop(columns=year_cols)
                    df[f'mrv_{current_year}'] = pd.to_numeric(last_vals, errors='coerce')
                    df = df.dropna()

            if as_frame:
                return df
                
            return df.reset_index().to_json(orient='records')

        except Exception as e:
            logger.error(f"Data retrieval failed: {e}")
            return pd.DataFrame() if as_frame else f"Error: {e}"

# --- PUBLIC API (USER FACING) ---

def search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search for World Bank indicators using natural language and Smart Ranking.
    
    Args:
        query: Search term (e.g., "Inflation", "education spending")
        limit: Maximum number of results.
    """
    logger.info(f"Searching for: '{query}'")
    
    async def _do_search():
        async with Data360Client() as client:
            search_engine = SearchEngine(client)
            raw = await search_engine.semantic_explore(query, database_id="WB_WDI")
            # Format for humans
            return [
                {
                    "code": item.get('series_description', {}).get('idno'),
                    "name": item.get('series_description', {}).get('name'),
                    "source": item.get('series_description', {}).get('database_id')
                }
                for item in raw[:limit]
            ]
        
    return _run_sync(_do_search())

def get_data(
    indicator: Union[str, List[str]], 
    economies: Union[str, List[str]], 
    years: int = 5,
    labels: bool = False,  # Convert codes to names
    as_json: bool = False
) -> Union[pd.DataFrame, str]:
    """
    Descarga datos corregidos y listos para usar (DataFrame por defecto).
    
    Args:
        indicator: Código(s) del indicador (ej: "NY.GDP.MKTP.KD" o "GDP")
        economies: Código(s) de país (ej: ["CHL", "PER"] o "USA")
        years: Número de años hacia atrás (default: 5). Si es 1, busca el valor más reciente (MRV).
        labels: Si es True, convierte códigos ISO (USA) a nombres legibles (United States).
               Preserva códigos originales en columna REF_AREA_CODE para compatibilidad con mapas.
        as_json: Si es True, Returns un string JSON en vez de DataFrame.
        
    Returns:
        pd.DataFrame (index=[REF_AREA, INDICATOR], columns=[Years...])
        o String JSON si as_json=True
    """
    logger.info(f"Fetching data via Senior Analyst Engine for {economies}...")
    
    # 1. Fetch Raw Data using internal async logic
    # Force as_frame=True so we can standardize. If user wants JSON, we convert at end.
    df = _run_sync(_async_get_data(indicator, economies, years, labels=labels, as_frame=True))
    
    if isinstance(df, str): # Error message
        return df
        
    # --- SENIOR ANALYST LAYER ---
    try:
        # 2. Standardize
        df = DataStandardizer.ensure_tidy(df)
        
        # 3. Audit
        report = DataAuditor.audit(df)
        DataAuditor.print_report(report)
    except Exception as e:
        # Fail safe: if analyst crashes, return raw data but log warning
        logger.warning(f"Senior Analyst Layer error: {e}")
    
    if as_json:
        return df.to_json(orient='records')
        
    return df

# Helper for lazy loading visualization
_viz_instance = None
def _get_viz():
    global _viz_instance
    if _viz_instance is None:
        try:
            from wbgapi360.visual.charts import Visualizer
            _viz_instance = Visualizer()
        except ImportError as e:
            if "seaborn" in str(e) or "matplotlib" in str(e):
                raise ImportError(
                    "Optional dependency 'seaborn' or 'matplotlib' not found. "
                    "Please install with: pip install wbgapi360[visual]"
                ) from e
            raise e
    return _viz_instance

def plot(chart_type: str, data: Union[str, pd.DataFrame], title: str = "", subtitle: str = "", **kwargs) -> str:
    """
    Generate Financial Times-style chart with editorial aesthetics.
    
    Args:
        chart_type: Chart type (trend, bar, scatter, map, map_bubble, map_diverging, map_categorical, etc.)
        data: DataFrame or JSON string with data
        title: Chart title
        subtitle: Chart subtitle
        **kwargs: Additional chart-specific arguments
                  (e.g., bins, labels for map_categorical)
        
    Returns:
        Absolute path to generated image.
    """
    logger.info(f"Plotting chart type: {chart_type}")
    
    # Lazy load visualizer
    viz = _get_viz()
    
    # Logic adapted for local dispatch
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
    
    func = dispatch_table.get(chart_type)
    if not func:
        raise ValueError(f"Unknown chart type: {chart_type}")
    
    # Handle JSON input
    if isinstance(data, str):
        try:
            import pandas as pd
            import json
            data = pd.DataFrame(json.loads(data))
        except Exception as e:
            raise ValueError(f"Invalid JSON data: {e}")
    
    # Generatetete unique path
    import tempfile, uuid
    chart_id = str(uuid.uuid4())[:8]
    path = os.path.join(tempfile.gettempdir(), f"wbg_plot_{chart_id}_{chart_type}.png")
    
    # Execute the plot function with all kwargs
    func(data, title=title, subtitle=subtitle, save_path=path, **kwargs)
    
    # Auto-display in Notebooks (Colab/Jupyter)
    try:
        from IPython.display import Image, display
        display(Image(filename=path))
    except (ImportError, Exception):
        pass

    return path


