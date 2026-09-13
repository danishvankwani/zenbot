from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


PDF_PATH = "data/AI18-EDA-Revision-Session.pdf"


loader = PyPDFLoader(PDF_PATH)

documents = loader.load()

print(f"Loaded {len(documents)} pages.")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")


print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


print("Creating FAISS vector store...")

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector store created.")


vectorstore.save_local(
    "faiss_index"
)

print("FAISS index saved.")