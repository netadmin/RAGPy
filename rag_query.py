import os
import sys
import argparse

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough


def main():
    parser = argparse.ArgumentParser(
        description="Query your Chroma RAG index with Ollama LLaMA3."
    )
    parser.add_argument(
        "-l", "--library-path", required=True,
        help="Root folder of your docs (e.g. D:/TivoRAG)"
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=10,
        help="Number of chunks to retrieve per query"
    )
    args = parser.parse_args()

    VECTOR_DIR = os.path.join(args.library_path, "vector_index_chroma")
    MODEL_NAME = "llama3"
    TOP_K = args.top_k

    print(f"🔌 Loading Chroma store from: {VECTOR_DIR}")
    embed_fn = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=embed_fn
    )
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

    print(f"🤖 Initializing LLaMA3 via OllamaLLM")
    llm = OllamaLLM(model=MODEL_NAME)

    # Define the prompt template
    prompt = PromptTemplate.from_template(
    """You are an expert TiVo support specialist who helps users troubleshoot their TiVo devices and services.
Your knowledge is based on official TiVo documentation. Be helpful, accurate, and concise.

When providing support:
- Prioritize step-by-step troubleshooting when applicable
- Include specific menu paths and button sequences when relevant
- Reference model-specific information when available in the context
- Explain technical terms in simple language

=== CONTEXT INFORMATION ===
{context}

=== USER QUESTION ===
{question}

=== RESPONSE ===
"""
    )

    print("🔗 Building RAG chain with modern runnable approach...")
    # Define the RAG chain using the modern runnable approach
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    print("\n✅ Ready! Type your query or 'exit' to quit.\n")
    while True:
        query = input("❓ Query> ").strip()
        if query.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break
        if not query:
            continue

        try:
            # Directly invoke the chain with the query
            answer = rag_chain.invoke(query)
            
            # Get source documents (this requires a separate call with the modern approach)
            sources = retriever.invoke(query)
            print("\n🔍 Raw retrieved content:")
            for i, doc in enumerate(sources):
                print(f"\n--- Document {i+1} ---")
                print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                
            print("\n🧠 Answer:\n" + answer)
            print("\n📚 Sources:")
            for doc in sources:
                print(f"  • {doc.metadata.get('source', 'Unknown')}")
                
        except Exception as e:
            print(f"❌ Query error: {e}")
            continue

if __name__ == "__main__":
    main()