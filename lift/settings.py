from dotenv import find_dotenv
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    # Paths. SCHEMA_DIR lives inside the package so the bundled schemas ship
    # with the wheel; override it (env or local.env) to point at your own library.
    PACKAGE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    SCHEMA_DIR: str = os.path.join(PACKAGE_DIR, "schemas")
    IMAGE_DPI: int = 96
    MIN_PDF_IMAGE_DIM: int = 692
    MIN_IMAGE_DIM: int = 692
    MODEL_CHECKPOINT: str = "datalab-to/lift-oss-0.1.7"
    TORCH_DEVICE: str | None = None
    MAX_OUTPUT_TOKENS: int = 12384
    TORCH_ATTN: str | None = None

    # vLLM server settings
    VLLM_API_KEY: str = "EMPTY"
    VLLM_API_BASE: str = "http://localhost:8000/v1"
    VLLM_MODEL_NAME: str = "lift"
    VLLM_GPUS: str = "0"
    MAX_VLLM_RETRIES: int = 6

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()