"""Settings read from environment / .env (never hard-code keys)."""
import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

DEFAULT_ORIGINS = "http://localhost:3000,http://localhost:5173"


@dataclass(frozen=True)
class Settings:
    api_key: str = ""                      # key clients must send in x-api-key
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-haiku-4-5"
    use_mock: bool = True
    max_input_tokens: int = 4000
    cors_origins: tuple = ("http://localhost:3000", "http://localhost:5173")
    usage_log_path: str = "logs/usage.jsonl"


@lru_cache
def get_settings() -> Settings:
    openai_key = os.getenv("OPENAI_API_KEY", "")
    mock_env = os.getenv("USE_MOCK_LLM", "").strip().lower()
    # explicit flag wins; otherwise fall back to mock when no key is present
    use_mock = mock_env in ("1", "true", "yes") if mock_env else not openai_key

    origins = tuple(o.strip() for o in os.getenv("CORS_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip())
    if len(origins) > 2:
        raise ValueError("CORS_ORIGINS must list at most 2 origins")

    return Settings(
        api_key=os.getenv("API_KEY", ""),
        openai_api_key=openai_key,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        use_mock=use_mock,
        max_input_tokens=int(os.getenv("MAX_INPUT_TOKENS", "4000")),
        cors_origins=origins,
        usage_log_path=os.getenv("USAGE_LOG", "logs/usage.jsonl"),
    )
