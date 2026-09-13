from rag import create_rag_chain


ask = create_rag_chain()


question = input(
    "\nAsk a question about the document: "
)


answer, documents = ask(question)


print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

print(answer)


print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

for i, document in enumerate(
    documents,
    start=1
):

    print(
        f"Source {i} | "
        f"Page {document.metadata.get('page', 'Unknown')}"
    )