import httpx
import logging
import asyncio
import time
from typing import Optional, Dict, Any, Union
from .models import Data360Response
from ..config import settings

# Configure structured logging (JSON simulation for Enterprise)
class JsonFormatter(logging.Formatter):
    def format(self, record):
        import json
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module
        }
        return json.dumps(log_obj)

logger = logging.getLogger("wbgapi360")
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(settings.LOG_LEVEL)

class CacheEntry:
    def __init__(self, data: Any, ttl: int):
        self.data = data
        self.expiry = time.time() + ttl

class Data360Client:
    """
    Async-first client for World Bank Data360 API.
    Enterprise Grade: Configurable, Cached, Retriable.
    """
    
    def __init__(self):
        self.base_url = settings.API_URL
        self.timeout = settings.TIMEOUT
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"}
        )
        
        # --- Enterprise Disk Cache (SQLite) ---
        import sqlite3
        import os
        from pathlib import Path
        
        cache_dir = Path(settings.CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_dir / "cache.db"
        
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        # Simple Key-Value Store with Expiry
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS requests (
                key TEXT PRIMARY KEY,
                value TEXT,
                expiry REAL
            )
        ''')
        self.conn.commit()
    
    def _get_from_db(self, key: str) -> Optional[Dict]:
        self.cursor.execute("SELECT value, expiry FROM requests WHERE key = ?", (key,))
        row = self.cursor.fetchone()
        if row:
            val, expiry = row
            if time.time() < expiry:
                import json
                return json.loads(val)
            else:
                # Lazy cleanup
                self.cursor.execute("DELETE FROM requests WHERE key = ?", (key,))
                self.conn.commit()
        return None

    def _save_to_db(self, key: str, data: Any, ttl: int):
        import json
        expiry = time.time() + ttl
        val = json.dumps(data)
        self.cursor.execute("INSERT OR REPLACE INTO requests (key, value, expiry) VALUES (?, ?, ?)", (key, val, expiry))
        self.conn.commit()

    async def close(self):
        await self.client.aclose()
        
    def _get_cache_key(self, method, endpoint, kwargs) -> str:
        # Simple string representation key
        import hashlib
        raw = f"{method}|{endpoint}|{str(kwargs)}"
        return hashlib.md5(raw.encode()).hexdigest()

    async def _request(self, method: str, endpoint: str, retries: int = settings.MAX_RETRIES, **kwargs) -> Dict[str, Any]:
        """Internal request handler with error management, retries, and caching."""
        
        # 1. Check Cache (Disk)
        if settings.ENABLE_CACHE and method == "GET":
            key = self._get_cache_key(method, endpoint, kwargs)
            cached_data = self._get_from_db(key)
            if cached_data:
                logger.debug(f'{{"event": "cache_hit", "endpoint": "{endpoint}", "source": "disk"}}')
                return cached_data

        for attempt in range(retries + 1):
            try:
                start_time = time.time()
                response = await self.client.request(method, endpoint, **kwargs)
                duration = time.time() - start_time
                
                response.raise_for_status()
                data = response.json()
                
                # 2. Update Cache (Disk)
                if settings.ENABLE_CACHE and method == "GET":
                     key = self._get_cache_key(method, endpoint, kwargs)
                     self._save_to_db(key, data, settings.CACHE_TTL)

                # Log success metrics
                logger.debug(f'{{"event": "request_success", "duration_ms": {duration*1000:.2f}}}')
                
                return data

            except httpx.HTTPStatusError as e:
                # Retry on 429 (Too Many Requests) or 5xx (Server Errors)
                if e.response.status_code == 429 or e.response.status_code >= 500:
                    if attempt < retries:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s...
                        logger.warning(f'{{"event": "retry", "reason": "status_{e.response.status_code}", "wait_s": {wait_time}}}')
                        await asyncio.sleep(wait_time)
                        continue
                logger.error(f'{{"event": "request_error", "status": {e.response.status_code}, "body": "{e.response.text}"}}')
                raise
            except (httpx.RequestError, httpx.TimeoutException) as e:
                if attempt < retries:
                     wait_time = 2 ** attempt
                     logger.warning(f'{{"event": "retry", "reason": "network_error", "wait_s": {wait_time}}}')
                     await asyncio.sleep(wait_time)
                     continue
                logger.error(f'{{"event": "request_failed", "retries": {retries}, "error": "{str(e)}"}}')
                raise
            except Exception as e:
                logger.error(f'{{"event": "unexpected_error", "error": "{str(e)}"}}')
                raise

    async def get_data(self, endpoint: str, params: Dict[str, Any] = None) -> Union[Dict, list]:
        """Generic GET wrapper."""
        return await self._request("GET", endpoint, params=params)

    async def post_data(self, endpoint: str, json_body: Dict[str, Any]) -> Union[Dict, list]:
        """Generic POST wrapper."""
        return await self._request("POST", endpoint, json=json_body)

    # --- Synchronous Context Manager Support (Optional for scripts) ---
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
