import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Medical Assistant API"
    VERSION: str = "1.0.0"
    
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    
    # Pinecone VectorStore
    PINECONE_ENV: str = "us-east-1"
    PINECONE_INDEX_NAME: str = "medicalindex"
    
    # Set explicit environment variables locally for library access if needed
    def model_post_init(self, __context):
        if self.GOOGLE_API_KEY:
            os.environ["GOOGLE_API_KEY"] = self.GOOGLE_API_KEY
            
settings = Settings()
