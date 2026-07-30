import argparse
import os
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHROMA_DIRECTORY = Path(__file__).resolve().parents[1] / "chroma_db"
DB_DIR = str(
    Path(os.getenv("CHROMA_DB_DIRECTORY", str(DEFAULT_CHROMA_DIRECTORY))).resolve()
)


def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_deployment=os.environ.get(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
            "text-embedding-ada-002"),
        openai_api_version=os.environ.get(
            "AZURE_OPENAI_API_VERSION",
            "2023-05-15"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    )


def ingest_handbook(pdf_path: str, reset: bool = False):
    """Load ESI Handbook PDF into ChromaDB."""
    if reset and os.path.exists(DB_DIR):
        print(f"Resetting database at {DB_DIR}...")
        shutil.rmtree(DB_DIR)

    print(f"Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # Create vector store
    print("Ingesting into ChromaDB (this may take a moment)...")
    embeddings = get_embeddings()

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="esi_handbook"
    )

    print(f"Ingestion complete. Vector store saved to {DB_DIR}")


def query_handbook(query: str, k: int = 3):
    embeddings = get_embeddings()
    vector_store = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="esi_handbook"
    )

    results = vector_store.similarity_search(query, k=k)
    print(f"\nQUERY: {query}")
    print("-" * 40)
    for i, res in enumerate(results):
        print(f"RESULT {i + 1} (Source: Page {res.metadata.get('page')}):")
        print(res.page_content[:300] + "...")
        print("-" * 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESI Handbook RAG Setup")
    parser.add_argument("--pdf", type=str, help="Path to the ESI Handbook PDF")
    parser.add_argument("--query", type=str, help="Run a test query")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the database before ingestion")

    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    if args.pdf:
        ingest_handbook(args.pdf, reset=args.reset)

    if args.query:
        query_handbook(args.query)

    if not args.pdf and not args.query:
        print("Please provide --pdf <path> to ingest or --query <text> to search.")
