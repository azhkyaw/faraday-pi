from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    gen_url: str = "http://localhost:8080"
    embed_url: str = "http://localhost:8081"
    embed_dim: int = 384
    db_path: str = "data/faraday.sqlite"
    chunk_size: int = 1200       # characters
    chunk_overlap: int = 200     # characters
    top_k: int = 4
    max_tokens: int = 512

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            gen_url=os.environ.get("FARADAY_GEN_URL", cls.gen_url),
            embed_url=os.environ.get("FARADAY_EMBED_URL", cls.embed_url),
            db_path=os.environ.get("FARADAY_DB", cls.db_path),
        )
