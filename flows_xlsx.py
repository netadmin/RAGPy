import os
import pandas as pd
import torch
import argparse
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def generate_ipflow_doc(row):
    return (
        f"Order: {row['Order']}\n"
        f"FlowID: {row['FlowID']}\n"
        f"Source: {row['Src DC']} (Zone: {row['Src Sec Zone']}, Network: {row['Src Sec Zone Network']})\n"
        f"Destination: {row['Dst DC']} (Zone: {row['Dst Sec Zone']}, Network: {row['Dst Sec Zone Network']})\n"
        f"Source Service: {row['Source Service']} | Destination Service: {row['Destination Service']}\n"
        f"Protocol: {row['Protocol']}, Port: {row['Port']}\n"
        f"Application: {row['Application']}\n"
        f"Usage: {row['Usage Detail']}\n"
        f"Active: {row['Active']} | Flag: {row['Flag']} | Tag: {row['Tag']}\n"
        f"Phase Impact: {row['PhaseImpact']}\n"
        f"UML Diagram Flow Text: {row['UML Diagram Flow Text']}\n"
        f"Src Note: {row['UML- Add IP Note Src']} | Dst Note: {row['UML - Add IP Note Dst']}\n"
        f"IP Flow: {row['UML IP Flow']}"
    ).strip()

def generate_whitelist_doc(row):
    return (
        f"Whitelist Site: {row['Site']}\n"
        f"IP Range: {row['Range']}\n"
        f"Usage: {row['Usage']}"
    ).strip()

def load_change_log(sheet):
    text = "\n".join(
        " | ".join(str(item) for item in row if pd.notna(item))
        for row in sheet.values
    )
    return Document(page_content=f"Change Log Entries:\n{text}", metadata={"sheet": "Change Log", "source_type": "excel_flows"})

def main():
    parser = argparse.ArgumentParser(description="Ingest IP Flows Excel data into a Chroma vector DB (non-destructive for PDFs/other sources).")
    parser.add_argument("--excel-path", "-e", required=True, help="Path to the Excel file (e.g. Flows 2025-01-22.xlsx)")
    parser.add_argument("--db-dir", "-d", required=True, help="Path to the Chroma vector DB directory")
    parser.add_argument("--model", "-m", default="BAAI/bge-large-en-v1.5", help="Embedding model to use")
    args = parser.parse_args()

    excel_path = args.excel_path
    vector_db_dir = args.db_dir
    embedding_model = args.model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device}
    )

    print(f"🔌 Opening Chroma store at {vector_db_dir}")
    store = Chroma(persist_directory=vector_db_dir, embedding_function=embedder)

    # 1. Delete previous Excel flow docs only (robustly)
    print("🧹 Deleting previous Excel-ingested flow docs...")
    try:
        results = store._collection.get(where={"source_type": "excel_flows"})
        excel_doc_ids = results.get("ids", [])
    except Exception as e:
        print(f"Error during fetching Excel doc IDs: {e}")
        excel_doc_ids = []

    if excel_doc_ids:
        store._collection.delete(ids=excel_doc_ids)
        print(f"🗑 Deleted {len(excel_doc_ids)} previous Excel docs.")
    else:
        print("🗑 No previous Excel docs found to delete.")

    # 2. Ingest new Excel data
    xls = pd.ExcelFile(excel_path)
    sheets = xls.sheet_names
    docs = []

    if "IPFlows" in sheets:
        df = xls.parse("IPFlows")
        for _, row in df.iterrows():
            doc = Document(
                page_content=generate_ipflow_doc(row),
                metadata={
                    "sheet": "IPFlows",
                    "FlowID": row.get("FlowID"),
                    "Order": row.get("Order"),
                    "source_type": "excel_flows"
                }
            )
            docs.append(doc)
        print(f"✅ Prepared {len(df)} flows from 'IPFlows'")

    # Accept "whitelist" sheet in any case (case-insensitive)
    whitelist_name = None
    for s in sheets:
        if s.lower() == "whitelist":
            whitelist_name = s
            break
    if whitelist_name:
        df = xls.parse(whitelist_name)
        for _, row in df.iterrows():
            doc = Document(
                page_content=generate_whitelist_doc(row),
                metadata={
                    "sheet": "whitelist",
                    "Site": row.get("Site"),
                    "Range": row.get("Range"),
                    "source_type": "excel_flows"
                }
            )
            docs.append(doc)
        print(f"✅ Prepared {len(df)} whitelist entries")

    if "Change Log" in sheets:
        sheet = xls.parse("Change Log")
        doc = load_change_log(sheet)
        docs.append(doc)
        print(f"✅ Prepared change log with {sheet.shape[0]} rows")

    print(f"📝 Total new Excel docs: {len(docs)}")

    # 3. Add new Excel docs to vector DB
    if docs:
        store.add_documents(docs)
        print("✅ Excel sheets updated in vector DB! (PDF/DOCX/etc untouched)")
    else:
        print("⚠️ No new Excel docs to add.")

if __name__ == "__main__":
    main()
