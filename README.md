# wbgapi360: World Bank Data Client for Humans & AI

[![PyPI version](https://img.shields.io/badge/pypi-v0.2.1-0052CC.svg)](https://pypi.org/project/wbgapi360/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-grey.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/MaykolMedrano/mcp_wbgapi360/actions/workflows/ci.yml/badge.svg)](https://github.com/MaykolMedrano/mcp_wbgapi360/actions)
[![Status](https://img.shields.io/badge/status-enterprise%20ready-green)]()

**wbgapi360** is a high-performance Python client designed to interact with the World Bank's Data APIs. It bridges the gap between traditional econometric analysis and modern Artificial Intelligence workflows.

This library is engineered with a **Hybrid Architecture**: it provides a synchronous, blocking API for human analysts (similar to `pandas` or `matplotlib`) and an asynchronous Model Context Protocol (MCP) server for AI Agents (such as Claude or OpenAI).

---

## Project Philosophy & Attribution

### Unofficial Client
This software is an independent open-source project. It is **not affiliated, associated, authorized, endorsed by, or in any way officially connected with The World Bank Group**. The official World Bank website can be found at [worldbank.org](https://www.worldbank.org).

### Inspiration & Legacy
This project is built upon the conceptual foundation laid by **Tim Herzog (tgherzog)** and his pioneering work with the original `wbgapi`. 

While `wbgapi` set the standard for Pythonic access to World Bank data, `wbgapi360` was built from the ground up to address two new challenges in the modern era:
1.  **AI Integration:** Native support for LLM function calling via MCP.
2.  **Smart Search:** Heuristic ranking algorithms to disambiguate indicators (e.g., distinguishing "GDP" from "Education Expenditure as % of GDP").

We acknowledge and thank Tim for establishing the developer-friendly patterns that inspired this library's human interface.

---

## Installation

Install using pip:

```bash
pip install wbgapi360
```

**For Visualization Features (Plots & Maps):**
The visualization module is optional to keep the library lightweight. To use `wb.plot()`, install with extras:
```bash
pip install "wbgapi360[visual]"
```
# With Visualization support (adds Matplotlib/Seaborn)
pip install wbgapi360[visual]

# With Mapping support (adds GeoPandas - Heavy dependency)
pip install wbgapi360[map]

---

## Usage Guide: For Analysts (Python)

The human interface is designed to be intuitive, synchronous, and safe for Jupyter Notebooks.

### 1. Smart Search (Semantic)
Finding the right indicator among 16,000+ options is difficult. `wbgapi360` uses a heuristic ranking system.

```python
import wbgapi360 as wb

# Search for "inflation" - automatically ranks "Consumer Price Index" above obscure sub-metrics
results = wb.search("inflation") # Returns top 10 matches

for r in results[:3]:
    print(f"[{r['code']}] {r['name']}")
# 1. Fetch GDP Growth (Last 10 years)
df = wb.get_data(
    indicator="NY.GDP.MKTP.KD.ZG",
    economies=["USA", "CHN", "PER"],
    years=10
)
print(df.head())

# 2. Fetch FDI Data (Last 20 years)
fdi_df = wb.get_data(
    indicator="BX.KLT.DINV.CD.WD",
    economies=["CHL", "USA", "CHN"],
    years=20
)
print("Foreign Direct Investment Data:")
print(fdi_df.head())

# 3. Plot Trend (Financial Times Style)
# Requires: pip install wbgapi360[visual]
data = wb.get_data("NY.GDP.PCAP.CD", ["CHL", "PER", "COL", "MEX"], years=15)

wb.plot(
    chart_type="trend",
    data=data,
    title="GDP Per Capita Trend",
    subtitle="USD Current"
)
```

### 4. Advanced Analysis: Trends & Statistics
Quickly assess the economic trajectory of a country without writing boilerplate pandas code.

```python
# Analyze the trend of Foreign Direct Investment in Chile
stats = wb.analyze_trend(
    indicator="BX.KLT.DINV.CD.WD", 
    economy="CHL", 
    years=20
)

# Returns a rich JSON with growth rates, volatility, and direction
# {
#   "statistics": {
#     "avg_annual_growth_pct": 5.2,
#     "volatility": 1240000000,
#     "trend_direction": "increasing"
#   }
# }
```

### 4. Professional Visualization
Generate Financial Times-style charts ready for publication.

```python
# 1. Fetch Data
data = wb.get_data("NY.GDP.PCAP.CD", ["CHL", "PER", "COL", "MEX"], years=15)

# 2. Plot Trend
wb.plot(
    chart_type="trend", 
    data=data, 
    title="Pacific Alliance: GDP per Capita",
    subtitle="Current US$ (2010-2025)",
    save_path="pacific_gdp.png"
)
```

---

## Usage Guide: For AI Agents (MCP)

This library implements the **Model Context Protocol (MCP)**, allowing AI agents (like Claude or Custom GPTs) to autonomously interact with World Bank data using high-level tools.

### 1. Configuration (Claude Desktop)
Add the following to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "worldbank": {
      "command": "uv",
      "args": [
        "--directory",
        "C:/path/to/mcp_wbgapi360",
        "run",
        "wbgapi360"
      ]
    }
  }
}
```

### 2. Available Tools
The Agent receives a toolkit optimized for economic reasoning:

| Tool | Description | Signature |
| :--- | :--- | :--- |
| `search_indicators` | Semantic search. Finds the best matching indicator codes. | `(query: str) -> List[Indicator]` |
| `get_data` | Fetches time-series data for analysis. | `(indicator_code: str, economies: List[str], years: int)` |
| `compare_countries` | Comparison logic. Fetches and normalizes data for multiple countries. | `(economies: List[str], indicators: List[str])` |
| `analyze_trend` | Returns statistical descriptive metadata (growth, volatility). | `(indicator: str, economy: str)` |
| `rank_countries` | Returns a sorted leaderboard for a given indicator. | `(indicator: str, region: str, top_n: int)` |
| `plot_chart` | Generates a visualization file from data. | `(chart_type: str, data: json)` |

### 3. Example Prompts
Once connected, you can ask the Agent complex economic questions:

> **"Compare the GDP growth volatility between Peru and Chile over the last 20 years. Who has been more stable?"**
> *   *Agent Action:* Calls `analyze_trend` for both countries and compares the `volatility` metric.

> **"Create a bar chart ranking the top 10 Latin American countries by Inflation."**
> *   *Agent Action:* Calls `rank_countries(indicator="FP.CPI.TOTL.ZG", region="latam")` then `plot_chart(chart_type="bar", ...)`.

> **"Is there a correlation between Literacy Rate and GDP per Capita in Africa?"**
> *   *Agent Action:* Calls `compare_countries` with both indicators for African nations, then `plot_chart(chart_type="scatter")`.

---

## Visualization Gallery

The library supports specific chart types optimized for economic data:

| Type | Description | Use Case |
| :--- | :--- | :--- |
| **Trend** | Multi-line time series | GDP Growth over time |
| **Bar** | Horizontal ranked bars | Comparing GINI index across countries |
| **Column** | Vertical categorical bars | Exports by region |
| **Scatter** | Correlation plot | Life Expectancy vs. GDP per Capita |
| **Map** | Choropleth map | Global population density (Requires geopandas) |

---

## Author

**Maykol Medrano**  
*Applied Economist Policy Data Scientist*  
Email: [mmedrano2@uc.cl](mailto:mmedrano2@uc.cl)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
