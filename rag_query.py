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
        "-k", "--top-k", type=int, default=20,
        help="Number of chunks to retrieve per query"
    )
    args = parser.parse_args()

    VECTOR_DIR = os.path.join(args.library_path, "vector_index_chroma")
    
    TOP_K = args.top_k

    print(f"🔌 Loading Chroma store from: {VECTOR_DIR}")
    embed_fn = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"  # Match the model used in ingest
    )
    db = Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=embed_fn
    )

   
    #MODEL_NAME = "llama3"
    #llm = OllamaLLM(
    #    model=MODEL_NAME,
    #    temperature=0.3,  # Lower temperature for more precise answers
    #    top_p=0.9,        # Keep high-probability tokens
    #    stop=["\n\n\n"],  # Prevent early stopping
    #    num_ctx=4096,        # Maximize context window utilization
    #    system="You are a helpful assistant that provides comprehensive and detailed answers. Use multiple paragraphs when needed."
    #)
    MODEL_NAME = "qwen3:4b" # or "qwen:14b" or "qwen2:7b"
    llm = OllamaLLM(
        model=MODEL_NAME,
        temperature=0.2,     # Lower temperature for Qwen works well for documentation
        top_p=0.95,          # Slightly higher for more comprehensive content
        stop=["\n\n\n\n"],   # Less restrictive stopping
        num_ctx=4096,
        system="You are a technical support specialist who provides exceptionally detailed, thorough answers. Always include all relevant information from documentation. Be comprehensive and use multiple paragraphs to fully explain concepts."
    )
    print(f"🤖 Loaded {MODEL_NAME} via OllamaLLM")
     
    # Define the prompt template
    prompt = PromptTemplate.from_template(
        """You are an expert TiVo support specialist and solution architect integrator who helps customers troubleshoot their TiVo IPTV Service.
        Your knowledge is based on official TiVo documentation. Be thorough, detailed, and comprehensive.

        When providing support:
        - Provide detailed step-by-step troubleshooting instructions
        - Include specific menu paths and button sequences when relevant
        - Reference model-specific information when available in the context
        - Explain technical concepts completely with examples where possible
        - Include all relevant information from the documentation
        - When you find multiple solutions in the documentation, include all of them
        - Use bullet points and numbered lists to organize longer explanations
        - When referencing sources, use the document title or filename, not internal IDs or UUIDs
        - Always provide exhaustive information - never summarize or abbreviate content from the documentation
        - Include contextual explanations for technical terms
        - Format your answer with clear sections and subsections
        - Remember that users need complete information, so provide all details from the documentation

        === CONTEXT INFORMATION ===
        {context}

        === USER QUESTION ===
        {question}

        === DETAILED RESPONSE ===
        """
    )

    print("🔗 Building RAG chain with modern runnable approach...")
    # V2 Define the RAG chain using the modern runnable approach
    def get_context_and_store(query):
        nonlocal retrieved_docs
        docs = get_retriever(query, db, TOP_K)
        retrieved_docs = docs
        
        # Format context with richer document information
        formatted_context = []
        for i, doc in enumerate(docs):
            title = doc.metadata.get('title', os.path.basename(doc.metadata.get('source', 'Unknown')))
            date_info = doc.metadata.get('date', '').split('T')[0] if 'date' in doc.metadata else ''
            
            # Format each document with clear boundaries and metadata
            context_entry = f"--- DOCUMENT {i+1}: {title} {f'({date_info})' if date_info else ''} ---\n"
            context_entry += f"{doc.page_content}\n"
            
            # Include relevance score information
            score = doc.metadata.get('score', doc.metadata.get('combined_score', 0))
            if score:
                # Only add this internally - helpful for debugging but not needed for the final prompt
                # context_entry += f"[Relevance: {score:.2f}]\n"
                pass
                
            formatted_context.append(context_entry)
        
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
