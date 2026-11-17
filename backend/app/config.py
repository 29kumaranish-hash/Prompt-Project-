from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / '.env', extra='ignore')
    ollama_base_url: str = 'http://localhost:11434'
    ollama_model: str = 'qwen2.5:3b'
    compression_model: str = 'microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank'
    compression_device: str = 'auto'
    default_compression_rate: float = Field(0.5, ge=0.1, le=1)
    chunk_target_tokens: int = Field(400, ge=64, le=450)
    max_upload_size_mb: int = Field(30, ge=1, le=200)
    max_pages: int = Field(500, ge=1)
    database_url: str = f'sqlite:///{(ROOT / "storage/researchcompress.db").as_posix()}'
    storage_path: Path = ROOT / 'storage'
    enable_baseline_comparison: bool = False
    ollama_context: int = Field(8192, ge=4096, le=131072)
    ollama_output_tokens: int = Field(1200, ge=256, le=2048)
    ollama_timeout: int = Field(600, ge=10)
    cors_origins: list[str] = ['http://localhost:3000', 'http://127.0.0.1:3000']


settings = Settings()
settings.storage_path.mkdir(parents=True, exist_ok=True)
