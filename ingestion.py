import asyncio
import logging
import os
import sys
import ssl
from typing import List, Dict, Any

import certifi
from dotenv import load_dotenv
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAI, OpenAIEmbeddings # Corrected import
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


async def main():
    print("Main async function to orchestrate the ingestion process.")


if __name__ == "__main__":
    asyncio.run(main())