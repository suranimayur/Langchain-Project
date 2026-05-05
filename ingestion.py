import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def main():
    try:
        print("🚀 Starting ingestion...")

        loader = TextLoader(
            r"D:\Udemy\Build AI Agents with Langchain\medium_blog.txt",
            encoding="utf-8"
        )

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        texts = splitter.split_documents(docs)

        print(f"✅ Created {len(texts)} chunks")

        # FIXED MODEL
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-large"
        )

        PineconeVectorStore.from_documents(
            documents=texts,
            embedding=embeddings,
            index_name=os.getenv("INDEX_NAME")
        )

        print("🎉 Upload Success!")

    except Exception as e:
        print("❌ Error:", e)

if __name__ == "__main__":
    main()