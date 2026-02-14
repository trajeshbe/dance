"""
Application configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""

    # Application
    APP_NAME: str = "Dance Video Generator"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # API
    API_V1_PREFIX: str = "/api/v1"

    # Database
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "dance_generator"

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    MINIO_BUCKET_PHOTOS: str = "dance-photos"
    MINIO_BUCKET_VIDEOS: str = "dance-videos"
    MINIO_BUCKET_REFERENCES: str = "dance-references"
    MINIO_BUCKET_AUDIO: str = "dance-audio"
    MINIO_BUCKET_POSES: str = "pose-sequences"

    # Celery
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None

    @property
    def CELERY_BROKER(self) -> str:
        return self.CELERY_BROKER_URL or self.REDIS_URL

    @property
    def CELERY_BACKEND(self) -> str:
        return self.CELERY_RESULT_BACKEND or self.REDIS_URL

    # AI Models Configuration

    # Model storage paths
    MODEL_CACHE_DIR: str = "./models"

    # FOMM (First Order Motion Model)
    FOMM_CHECKPOINT_PATH: str = "./models/fomm/vox-256.pth"
    FOMM_CONFIG_PATH: str = "./models/fomm/vox-256.yaml"

    # LivePortrait
    LIVEPORTRAIT_CHECKPOINT_DIR: str = "./models/liveportrait"

    # MagicAnimate
    MAGICANIMATE_MODEL_PATH: str = "./models/magic-animate"

    # MagicDance
    MAGICDANCE_MODEL_PATH: str = "./models/magic-dance"

    # YOLOv8 for person detection
    YOLO_MODEL: str = "yolov8x.pt"

    # SAM for segmentation
    SAM_CHECKPOINT: str = "./models/sam/sam_vit_h_4b8939.pth"
    SAM_MODEL_TYPE: str = "vit_h"

    # GPU Configuration
    CUDA_VISIBLE_DEVICES: str = "0"
    USE_GPU: bool = True

    # Processing Configuration
    MAX_CONCURRENT_JOBS: int = 2  # Max parallel video generations
    VIDEO_OUTPUT_FPS: int = 30
    VIDEO_OUTPUT_RESOLUTION: int = 1024  # 1024x1024

    # Face Expression Configuration
    ENABLE_FACIAL_EXPRESSIONS: bool = True
    FACIAL_EXPRESSION_MODEL: str = "liveportrait"  # "fomm", "liveportrait", "face-vid2vid"
    ENABLE_LIP_SYNC: bool = True
    EXPRESSION_INTENSITY: float = 1.0  # 0.0 - 2.0

    # API Keys (for proprietary services)
    SEEDANCE_API_KEY: Optional[str] = None
    REPLICATE_API_TOKEN: Optional[str] = None

    # Google Colab Integration
    COLAB_WORKER_URL: Optional[str] = None
    PREFER_COLAB: bool = True

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:3001"]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
