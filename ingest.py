from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma

load_dotenv()

DOCS_DIR = "documents"
DB_DIR = "vector_db"


def load_documents():
    docs = []

    for file in os.listdir(DOCS_DIR):

        path = os.path.join(DOCS_DIR, file)

        print("Loading:", file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        elif file.endswith(".txt"):
            loader = TextLoader(path)
            docs.extend(loader.load())

    return docs


def main():

    print("Loading documents...")
    documents = load_documents()
    print(documents[0].page_content[:500])

    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    print("Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print("Vector database created successfully!")


if __name__ == "__main__":
    main()