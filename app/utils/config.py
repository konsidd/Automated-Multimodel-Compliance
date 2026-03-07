"""
app/utils/config.py
──────────────────
Centralised configuration loader.
Reads configs/settings.yaml and merges with .env overrides.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

load_dotenv()

# ── Root of the project (two levels up from this file) ──────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT_DIR / "configs" / "settings.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ── Pydantic sub-models ───────────────────────────────────────────────────────

class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "llama3"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 2048


class EmbeddingsConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"


class VectorStoreConfig(BaseModel):
    index_path: str = "data/vector_index/faiss.index"
    metadata_path: str = "data/vector_index/metadata.json"
    top_k: int = 5


class IngestionConfig(BaseModel):
    whisper_model: str = "base"
    temp_video_dir: str = "data/raw_videos"
    transcript_dir: str = "data/transcripts"
    ocr_lang: str = "eng"
    chunk_size: int = 512
    chunk_overlap: int = 64


class ComplianceConfig(BaseModel):
    policies_dir: str = "data/policies"
    rules_file: str = "configs/rules.yaml"
    score_threshold: float = 0.75


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True


class TelemetryConfig(BaseModel):
    log_level: str = "INFO"
    langsmith_project: str = "compliance-orchestrator"
    enable_tracing: bool = True


class AppConfig(BaseModel):
    name: str = "Compliance Orchestrator"
    version: str = "0.1.0"
    debug: bool = False

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    @property
    def root_dir(self) -> Path:
        return ROOT_DIR


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and cache config from YAML."""
    raw = _load_yaml(CONFIG_FILE)
    app_raw = raw.get("app", {})
    return AppConfig(
        name=app_raw.get("name", "Compliance Orchestrator"),
        version=app_raw.get("version", "0.1.0"),
        debug=app_raw.get("debug", False),
        llm=LLMConfig(**raw.get("llm", {})),
        embeddings=EmbeddingsConfig(**raw.get("embeddings", {})),
        vector_store=VectorStoreConfig(**raw.get("vector_store", {})),
        ingestion=IngestionConfig(**raw.get("ingestion", {})),
        compliance=ComplianceConfig(**raw.get("compliance", {})),
        api=APIConfig(**raw.get("api", {})),
        telemetry=TelemetryConfig(**raw.get("telemetry", {})),
    )
