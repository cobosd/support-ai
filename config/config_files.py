from dataclasses import dataclass
import streamlit as st

# load_dotenv(find_dotenv())
@dataclass(frozen=True)
class APIkeys:
    PineconeAPI: str = st.secrets['PINECONE_API']
    PineconeEnv: str = st.secrets['PINECONE_ENV_ADA']
    PineconeIdx: str = st.secrets['PINECONE_INDEX_ADA']
    OpenAiAPI: str = st.secrets['OPENAI_KEY']
    
@dataclass(frozen=True)
class ModelParams:
    model: str = ""
    temperature: str = ""
    usertitle: str = ""
    domain: str = ""
    embedding_model: str = ""