from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    "data/AI18-EDA-Revision-Session.pdf"
)

documents = loader.load()

print(f"Number of pages: {len(documents)}")

print("\nFirst page:\n")
print(documents[0].page_content[:2000])