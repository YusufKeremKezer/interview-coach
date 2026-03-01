from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    ASSEMBLYAI_API_KEY: str
    OPENROUTER_API_KEY: str
    SERPER_API_KEY: str
    COMPANY_NAME: str = "Google"
    ROLE: str = "AI Engineer"
    DIFFICULTY: str = "Medium"

settings = Settings()