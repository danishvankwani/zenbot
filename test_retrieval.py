from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


query = input(
    "\nAsk a question about the document: "
)

results = retriever.invoke(query)


print("\n" + "=" * 60)
print("RETRIEVED DOCUMENT CHUNKS")
print("=" * 60)


for i, document in enumerate(
    results,
    start=1
):

    print(f"\n--- Result {i} ---")

    print(
        f"Page: {document.metadata.get('page', 'Unknown')}"
    )

    print("\nContent:")

    print(document.page_content)