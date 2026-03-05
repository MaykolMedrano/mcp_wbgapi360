<div align="center">
  
# wbgapi360
  
**World Bank Data API client with two interfaces: A synchronous Python API for analysts, and a Model Context Protocol (MCP) server for AI agents.**

[![PyPI version](https://img.shields.io/pypi/v/wbgapi360.svg?style=flat-square&color=blue)](https://pypi.org/project/wbgapi360/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![CI](https://img.shields.io/github/actions/workflow/status/MaykolMedrano/mcp_wbgapi360/ci.yml?branch=master&style=flat-square)](https://github.com/MaykolMedrano/mcp_wbgapi360/actions/workflows/ci.yml)
[![Downloads](https://img.shields.io/pypi/dm/wbgapi360?style=flat-square&color=blue)](https://pypi.org/project/wbgapi360/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
  
</div>

---

## What You Get

- Smart indicator discovery (`search`) with heuristic ranking.
- High-level data retrieval (`get_data`) with code normalization.
- Publication-style charts (`plot`) with optional visual dependencies.
- MCP tools for search, data access, comparisons, trend analysis, rankings, and chart generation.

## Installation

```bash
pip install wbgapi360
```

Optional extras:

```bash
# Visualization support (matplotlib + seaborn)
pip install "wbgapi360[visual]"

# Mapping support (geopandas + matplotlib)
pip install "wbgapi360[map]"
```

## Python API (Analysts)

```python
import wbgapi360 as wb

# 1) Search indicators
results = wb.search("inflation")
for r in results[:3]:
    print(f"[{r['code']}] {r['name']}")

# 2) Fetch data
df = wb.get_data(
    indicator="NY.GDP.MKTP.KD.ZG",
    economies=["USA", "CHN", "PER"],
    years=10
)
print(df.head())

# 3) Plot (requires: pip install "wbgapi360[visual]")
wb.plot(
    chart_type="trend",
    data=df,
    title="GDP Growth",
    subtitle="Annual %"
)
```

## MCP Server (AI Agents)

### Run the server

After installation:

```bash
wbgapi360
```

From repository source:

```bash
python -m wbgapi360.mcp.run
```

### Claude Desktop example

```json
{
  "mcpServers": {
    "worldbank": {
      "command": "wbgapi360",
      "args": []
    }
  }
}
```

### MCP tools

| Tool | Description |
| :--- | :--- |
| `search_indicators` | Semantic search for indicator codes. |
| `get_data` | Retrieve time series data for one or many countries. |
| `compare_countries` | Compare countries across one or many indicators. |
| `analyze_trend` | Return trend statistics (growth, volatility, direction). |
| `rank_countries` | Rank countries for an indicator globally or by region. |
| `plot_chart` | Generate chart images from JSON payloads. |

## CLI (Developer Utility)

A separate CLI is provided for manual testing and quick queries:

```bash
wbgapi360-cli --help
```

Commands:

- `search`
- `data`
- `config`

## Development

```bash
pip install -e ".[dev,visual]"
python -m pytest -q tests
```

## Project Notes

### Unofficial client

This software is an independent open-source project. It is not affiliated with The World Bank Group.

### Attribution

`wbgapi360` is inspired by Tim Herzog's original `wbgapi` project and extends it with MCP support and AI-focused workflows.

## Author

Maykol Medrano  
Applied Economist / Policy Data Scientist  
Email: [mmedrano2@uc.cl](mailto:mmedrano2@uc.cl)

## License

MIT. See `LICENSE`.
