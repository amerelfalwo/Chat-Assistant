import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Chat Assistant API"
    VERSION: str = "1.0.0"

    GOOGLE_API_KEY: str = ""
    GROQ_API_KEY: str
    PINECONE_API_KEY: str
    
    PINECONE_ENV: str = "us-east-1"
    PINECONE_INDEX_NAME: str = "medicalindex"
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore" 
    )
    
    def model_post_init(self, __context):
        if self.GOOGLE_API_KEY:
            os.environ["GOOGLE_API_KEY"] = self.GOOGLE_API_KEY
        if self.GROQ_API_KEY:
            os.environ["GROQ_API_KEY"] = self.GROQ_API_KEY
        if self.PINECONE_API_KEY:
            os.environ["PINECONE_API_KEY"] = self.PINECONE_API_KEY
            
settings = Settings()