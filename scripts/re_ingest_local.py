from pathlib import Path
from backend.ingest import process_jsonl, ingest_to_chroma
from backend.config import get_settings
from langchain_core.documents import Document

BASE = Path(__file__).resolve().parent.parent / "docs_temp"

if __name__ == "__main__":
    settings = get_settings()
    all_chunks = []
    files = sorted(BASE.glob("*.jsonl"))
    if not files:
        print("No JSONL found in docs_temp/")
        raise SystemExit(1)

    for f in files:
        print(f"Processing {f.name}")
        chunks = process_jsonl(f, {})
        print(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"Total chunks to ingest: {len(all_chunks)}")
    if not all_chunks:
        print("Nothing to ingest, exiting.")
        raise SystemExit(1)

    ingest_to_chroma(chunks=all_chunks, settings=settings, clear_existing=True)
    print("Re-ingest local complete.")
