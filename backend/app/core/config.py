from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator


class Settings(BaseSettings):
    DATABASE_URL: str

    # App constraints
    MAX_FRAME_SIZE_MB: int = Field(default=10, gt=0)
    VIDEO_MAX_SIZE_MB: int = Field(default=500, gt=0)
    FRAME_PROCESSING_INTERVAL: int = Field(default=5, ge=1)
    FRAME_SAVE_INTERVAL: int = Field(default=30, ge=1)

    # ML Hyperparameters
    EAR_THRESHOLD: float = Field(default=0.20, gt=0, lt=1)
    MAR_THRESHOLD: float = Field(default=0.50, gt=0, lt=1)
    YAWN_MIN_DURATION_SEC: float = Field(default=1.5, gt=0)
    HEAD_PITCH_THRESHOLD_DEG: float = Field(default=20.0, gt=0)
    FACE_ABSENT_THRESHOLD_SEC: float = Field(default=3.0, gt=0)
    PERCLOS_WINDOW_SEC: float = Field(default=60.0, gt=0)
    PERCLOS_MILD: float = Field(default=0.35, gt=0, lt=1)
    PERCLOS_SEVERE: float = Field(default=0.70, gt=0, lt=1)
    BLINK_MAX_DURATION_MS: int = Field(default=400, gt=0)
    BLINK_MIN_DURATION_MS: int = Field(default=50, gt=0)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "Settings":
        if self.PERCLOS_MILD >= self.PERCLOS_SEVERE:
            raise ValueError(
                f"PERCLOS_MILD ({self.PERCLOS_MILD}) must be less than "
                f"PERCLOS_SEVERE ({self.PERCLOS_SEVERE})"
            )
        if self.BLINK_MIN_DURATION_MS >= self.BLINK_MAX_DURATION_MS:
            raise ValueError(
                f"BLINK_MIN_DURATION_MS ({self.BLINK_MIN_DURATION_MS}) must be "
                f"less than BLINK_MAX_DURATION_MS ({self.BLINK_MAX_DURATION_MS})"
            )
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """Returns a cached instance of settings."""
    return Settings()