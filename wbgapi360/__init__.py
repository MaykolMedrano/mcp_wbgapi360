from .core.client import Data360Client
from .search.engine import SearchEngine
from .data.builder import DataBuilder
from .ai.agent import DataAgent
from .metadata.builder import MetadataBuilder
from .metadata.builder import MetadataBuilder
# from .visual import viz # LAZY LOADED


class API:
    def __init__(self):
        self._client = None
    
    @property
    def client(self):
         if not self._client:
             self._client = Data360Client()
         return self._client
    
    @property
    def search(self):
        return SearchEngine(self.client)

    @property
    def data(self):
        return DataBuilder(self.client)

    @property
    def metadata(self):
        return MetadataBuilder(self.client)

    @property
    def ai(self):
        return DataAgent(self.client)

    @property
    def visual(self):
        try:
            from .visual import viz
            return viz
        except ImportError as e:
            if "seaborn" in str(e) or "matplotlib" in str(e):
                 raise ImportError(
                    "Optional dependency 'seaborn' not found. "
                    "Install with: pip install wbgapi360[visual]"
                ) from e
            raise e
        
    async def close(self):
        if self._client:
            await self._client.close()

__version__ = "0.2.8"
__author__ = "Maykol Medrano"
__email__ = "mmedrano2@uc.cl"
__credits__ = ["Applied Economist Policy Data Scientist"]

# Expose the human-friendly API at top level
from wbgapi360.api import search, get_data, plot

__all__ = ["search", "get_data", "plot"]
