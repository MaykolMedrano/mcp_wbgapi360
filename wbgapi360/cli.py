import argparse
import asyncio
import sys
import json
from . import api
from .config import settings

async def run_search(args):
    """Execute search command."""
    print(f"Searching for: {args.query}...")
    try:
        results = await api.search.semantic_explore(args.query)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"Found {len(results)} results:")
            for r in results[:10]:
                 desc = r.get('series_description', {})
                 print(f"[{desc.get('idno')}] {desc.get('name')}")
    except Exception as e:
        print(f"Error: {e}")

async def run_data(args):
    """Execute data fetch command."""
    print(f"Fetching data: {args.indicator} for {args.economy}...")
    try:
        data = await api.data.indicator(args.indicator).economy(args.economy).limit(args.limit).get()
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print(f"Returned {len(data)} rows.")
            # Simple table print for CLI
            print(f"{'Economy':<10} {'Year':<10} {'Value':<15}")
            print("-" * 35)
            for row in data:
                # Based on OAS, fields might be uppercase like OBS_VALUE
                eco = row.get('REF_AREA') or row.get('economy') or 'N/A'
                time = row.get('TIME_PERIOD') or row.get('time') or 'N/A'
                val = row.get('OBS_VALUE') or row.get('value') or 'N/A'
                print(f"{eco:<10} {time:<10} {val:<15}")

    except Exception as e:
        print(f"Error: {e}")

async def async_main():
    parser = argparse.ArgumentParser(description="wbgapi360 Enterprise CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Search Command
    search_parser = subparsers.add_parser("search", help="Semantic search for indicators")
    search_parser.add_argument("query", help="Search term or natural language query")
    search_parser.add_argument("--json", action="store_true", help="Output raw JSON")

    # Data Command
    data_parser = subparsers.add_parser("data", help="Fetch data")
    data_parser.add_argument("--indicator", required=True, help="Indicator ID")
    data_parser.add_argument("--economy", default="WLD", help="Economy code (default: WLD)")
    data_parser.add_argument("--limit", type=int, default=5, help="Limit results")
    data_parser.add_argument("--json", action="store_true", help="Output raw JSON")

    # Config Command
    config_parser = subparsers.add_parser("config", help="Show current configuration")

    args = parser.parse_args()

    if args.command == "search":
        await run_search(args)
    elif args.command == "data":
        await run_data(args)
    elif args.command == "config":
        print("Current Configuration:")
        print(settings.model_dump_json(indent=2))
    
    await api.close()

def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nAborted.")

if __name__ == "__main__":
    main()
