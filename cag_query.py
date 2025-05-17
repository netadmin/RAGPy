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
        description="CAG query with Qwen."
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

    MODEL_NAME = "qwen3:4b" # or "qwen:14b" or "qwen2:7b"
    llm = OllamaLLM(
        model=MODEL_NAME,
        temperature=0.2,     # Lower temperature for Qwen works well for documentation
        top_p=0.95,          # Slightly higher for more comprehensive content
        #stop=["\n\n\n\n"],   # Less restrictive stopping
        num_ctx=4096,
        system="Answer the user's question using ONLY the context provided. If the context does not answer, say 'Not found in context.'"
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
        
    print("\n✅ Ready! Type your query or 'exit' to quit.\n")
    while True:
        query = input("❓ Query> ").strip()
        if query.lower() in ("exit", "quit"): break
        if not query: continue

        # 1. Retrieve chunks
        docs = db.as_retriever(search_type="similarity", search_kwargs={"k": args.top_k}).invoke(query)
        
        # 2. Query Qwen per chunk
        answers = []
        for i, doc in enumerate(docs):
            ctx = doc.page_content
            filled_prompt = prompt.format(context=ctx, question=query)
            answer = llm.invoke(filled_prompt)
            answers.append((i+1, doc, answer.strip()))

        # 3. Display chunk-wise answers
        print("\n🧠 Chunk-level Answers:")
        for idx, doc, ans in answers:
            title = doc.metadata.get('title', os.path.basename(doc.metadata.get('source', 'Unknown')))
            print(f"\n--- Chunk {idx} ({title}) ---\n{ans}")

        # 4. Optional: "Synthesize" final answer (e.g., majority vote or send all answers to Qwen again for summary)
        print("\n🔗 Synthesized Final Answer (auto summary):")
        synthesized = llm.invoke("Summarize or synthesize the best possible answer from these:\n" +
                                 "\n".join(f"Chunk {i}: {a}" for i, _, a in answers))
        print(synthesized)

if __name__ == "__main__":
    main()
