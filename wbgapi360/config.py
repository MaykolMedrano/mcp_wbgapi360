import os
from pydantic import BaseModel, Field

class Settings(BaseModel):
    """
    Enterprise Configuration Management.
    Loads from Environment Variables tailored for Docker/K8s.
    """
    API_URL: str = Field(default="https://data360api.worldbank.org/data360", alias="WBG360_API_URL")
    TIMEOUT: int = Field(default=30, alias="WBG360_TIMEOUT")
    MAX_RETRIES: int = Field(default=3, alias="WBG360_MAX_RETRIES")
    ENABLE_CACHE: bool = Field(default=True, alias="WBG360_ENABLE_CACHE")
    CACHE_TTL: int = Field(default=300, alias="WBG360_CACHE_TTL") # 5 minutes default
    CACHE_DIR: str = Field(default=os.path.join(os.path.expanduser("~"), ".wbgapi360"), alias="WBG360_CACHE_DIR")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", alias="WBG360_LOG_LEVEL")

    @classmethod
    def load(cls):
        # Initial load from os.environ
        # Pydantic v2 usually uses pydantic-settings, but to avoid extra deps for this prototype step,
        # we manually populate from os.environ referencing the aliases if present.
        data = {}
        defaults = cls().model_dump()
        
        for name, field in cls.model_fields.items():
            alias = field.alias or name
            if alias in os.environ:
                val = os.environ[alias]
                # Simple boolean conversion
                if field.annotation is bool:
                    val = val.lower() in ('true', '1', 'yes')
                # Simple int conversion
                elif field.annotation is int:
                    val = int(val)
                data[name] = val
        
        return cls(**data)

settings = Settings.load()
