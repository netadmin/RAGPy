import os, argparse

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

def get_retriever(query, db, TOP_K):
    # Check if query implies time sensitivity
    time_sensitive = any(term in query.lower() for term in [
        "recent", "latest", "newest", "update", "current", 
        "today", "month", "year", "version"
    ])
    
    if time_sensitive:
        print("🕒 Time-sensitive query detected - prioritizing recent documents")
        # Use hybrid search with custom scoring
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K * 2}  # Get more results than needed
        )
        results = retriever.invoke(query)
        
        # Rerank results by combining semantic score with recency
        for doc in results:
            # Default recency score if missing
            recency = doc.metadata.get("recency_score", 0.5)
            # Combine semantic similarity with recency (70% similarity, 30% recency)
            combined_score = 0.7 * doc.metadata.get("score", 0) + 0.3 * recency
            doc.metadata["combined_score"] = combined_score
        
        # Sort by combined score
        results = sorted(results, key=lambda d: d.metadata.get("combined_score", 0), reverse=True)
        return results[:TOP_K]
    
    # Your existing retrieval logic
    elif any(term in query.lower() for term in ["error", "code", "message", "failed"]):
        print("🔍 Error message detected - using high precision retrieval")
        # Use similarity search with higher k to prioritize exact matches
        return db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K}
        ).invoke(query)
    else:
        print("🔍 General query - using balanced retrieval")
        return db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 2, "lambda_mult": 0.7}
        ).invoke(query)


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
        model_name="BAAI/bge-large-en-v1.5"  # Match the model used in ingest
    )
    db = Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=embed_fn
    )
    retriever = db.as_retriever(
        search_type="similarity",  # Use standard similarity search
        search_kwargs={
            "k": TOP_K
        }
    )

    print("🤖 Initializing LLaMA3 via OllamaLLM")
    llm = OllamaLLM(
        model=MODEL_NAME,
        temperature=0.2,  # Lower temperature for more precise answers
        top_p=0.9,        # Keep high-probability tokens
        stop=["\n\n\n"],  # Prevent early stopping
    )

    # Define the prompt template
    prompt = PromptTemplate.from_template(
    """You are an expert TiVo support specialist who helps users troubleshoot their TiVo devices and services.
    Your knowledge is based on official TiVo documentation. Be helpful, accurate, and concise.

    When providing support:
    - Prioritize step-by-step troubleshooting when applicable
    - Include specific menu paths and button sequences when relevant
    - Reference model-specific information when available in the context
    - Explain technical terms in simple language
    - When referencing sources, use the document title or filename, not internal IDs or UUIDs
    - If you're unsure about the exact source, simply state the information without referencing a specific document

    === CONTEXT INFORMATION ===
    {context}

    === USER QUESTION ===
    {question}

    === RESPONSE ===
    """
    )

    print("🔗 Building RAG chain with modern runnable approach...")
    # V2 Define the RAG chain using the modern runnable approach
    def get_context_and_store(query):
        nonlocal retrieved_docs
        docs = get_retriever(query, db, TOP_K)
        retrieved_docs = docs
        
        # Format context with document titles
        formatted_context = []
        for i, doc in enumerate(docs):
            # Get document title or filename as fallback
            title = doc.metadata.get('title', os.path.basename(doc.metadata.get('source', 'Unknown')))
            formatted_context.append(f"[Document: {title}]\n{doc.page_content}\n")
        
        return "\n".join(formatted_context)
        
    rag_chain = (
        {
            "context": lambda x: get_context_and_store(x),
            "question": RunnablePassthrough()
        }
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
            # Initialize storage for this query
            retrieved_docs = []
            # Directly invoke the chain with the query
            answer = rag_chain.invoke(query)

            # Get source documents (this requires a separate call with the modern approach)
            #sources = retriever.invoke(query)
            sources = retrieved_docs

            print("\n🧠 Answer:\n" + answer)
            print("\n📚 Sources:")
            # Track unique sources to avoid duplicates
            seen_sources = set()
            for doc in sources:
                source = doc.metadata.get('source', 'Unknown')
                date_info = ""
                if "date" in doc.metadata:
                    date_info = f" (dated: {doc.metadata['date'][:10]})"

                # Create a unique identifier for this source
                source_key = f"{source}{date_info}"

                # Only print if we haven't seen this source before
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    print(f"  • {source}{date_info}")

        except Exception as e:
            print(f"❌ Query error: {e}")
            continue

if __name__ == "__main__":
    main()
