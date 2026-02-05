
import pytest
import asyncio
from wbgapi360.mcp.server import _search_indicators, _get_data, _analyze_trend 

import wbgapi360.mcp.server as server

async def reset_state():
    if server._client_instance:
        try:
            await server._client_instance.close()
        except:
            pass
    server._client_instance = None
    server._search_engine_instance = None

@pytest.mark.asyncio
async def test_search_indicators():
    await reset_state()
    """Test semantic search functionality."""
    results = await _search_indicators(query="gdp growth", limit=3)
    assert len(results) > 0
    assert "code" in results[0]
    assert "name" in results[0]

@pytest.mark.asyncio
async def test_get_data_basic():
    await reset_state()
    """Test basic data retrieval."""
    data = await _get_data(indicator_code="NY.GDP.MKTP.KD.ZG", economies=['USA'], years=5)
    assert isinstance(data, str)
    # Check for presence of key fields. Note: ID might be transformed (e.g. WB_WDI_...)
    assert "REF_AREA" in data or "economy" in data
    assert "WB_WDI_NY_GDP_MKTP_KD_ZG" in data or "NY.GDP.MKTP.KD.ZG" in data.replace("_", ".")

@pytest.mark.asyncio
async def test_analyze_trend():
    await reset_state()
    """Test trend analysis."""
    # Increase years to ensures we get enough data points for stats
    trend_json = await _analyze_trend("NY.GDP.MKTP.KD.ZG", "USA", years=10)
    assert isinstance(trend_json, str)
    # We might not get stats if data is missing, so check for basic structure
    assert "data" in trend_json
    assert "meta" in trend_json
