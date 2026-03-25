"""Exporta una vista previa de los chunks generados para cada archivo.

Genera `reports/chunks_preview.txt` con: archivo, índice de chunk,
longitud y texto del chunk, y metadata (si aplica).

Este script evita importar todo `backend.ingest` para no requerir
las dependencias pesadas en tiempo de importación.
"""
from pathlib import Path
import json
from typing import List

# Reusar la misma configuración mínima que usa backend/ingest.py
LOCAL_DOCS_DIR = Path("./docs_temp")
INGEST_CONFIG_PATH = Path("./docs/ingest_guides/ingest_config.json")
REPORTS_DIR = Path("./reports")
OUTPUT_FILE = REPORTS_DIR / "chunks_preview.txt"

# Separadores en el mismo orden de prioridad definido en ingest.py
TEXT_SPLITTERS: List[str] = ["\n\n", "\n", ".", " ", ""]


def load_ingest_config() -> dict:
    if not INGEST_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(INGEST_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def recursive_split(text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []

    if not separators:
        if chunk_size <= 0:
            return [text]
        step = max(1, chunk_size - chunk_overlap)
        return [text[i:i+chunk_size].strip() for i in range(0, len(text), step)]

    sep = separators[0]
    if sep == "":
        return recursive_split(text, [], chunk_size, chunk_overlap)

    parts = text.split(sep)
    chunks: List[str] = []
    buffer = ""
    for i, part in enumerate(parts):
        if buffer:
            candidate = buffer + sep + part
        else:
            candidate = part

        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            if buffer:
                chunks.extend(recursive_split(buffer, separators[1:], chunk_size, chunk_overlap))
            # If single part is already too large, descend deeper
            buffer = part

    if buffer:
        chunks.extend(recursive_split(buffer, separators[1:], chunk_size, chunk_overlap))

    # Post-process to ensure sliding overlap on the produced chunks
    final: List[str] = []
    for c in chunks:
        if len(c) <= chunk_size:
            final.append(c.strip())
        else:
            step = max(1, chunk_size - chunk_overlap)
            for i in range(0, len(c), step):
                final.append(c[i:i+chunk_size].strip())

    # Remove empty items and return
    return [s for s in final if s]


def process_jsonl(path: Path) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            chunk_text = str(record.get("chunk_text", "") or "").strip()
            metadata = record.get("metadata", {}) if isinstance(record.get("metadata", {}), dict) else {}
            if chunk_text:
                items.append({"text": chunk_text, "metadata": metadata})
    return items


def process_plain_text(path: Path, file_config: dict) -> List[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []

    chunk_size = int(file_config.get("chunk_size", 300))
    chunk_overlap = int(file_config.get("chunk_overlap", 10))

    splits = recursive_split(text, TEXT_SPLITTERS, chunk_size, chunk_overlap)
    defaults = file_config.get("defaults", {})
    items = []
    for s in splits:
        items.append({"text": s, "metadata": defaults})
    return items


def main():
    cfg = load_ingest_config()
    if not LOCAL_DOCS_DIR.exists():
        print(f"Directorio {LOCAL_DOCS_DIR} no existe. Ejecuta la ingesta o coloca archivos en él.")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    out_lines = []
    total_chunks = 0

    for file_path in sorted(LOCAL_DOCS_DIR.iterdir()):
        if not file_path.is_file():
            continue
        name = file_path.name
        file_config = cfg.get(name, {})
        ext = file_path.suffix.lower()

        if ext == ".jsonl":
            items = process_jsonl(file_path)
        elif ext in {".txt", ".md", ".html"}:
            items = process_plain_text(file_path, file_config)
        else:
            continue

        out_lines.append("--- File: %s ---\n" % name)
        for i, it in enumerate(items, start=1):
            text = it.get("text", "").replace("\r\n", "\n")
            meta = it.get("metadata", {})
            out_lines.append(f"Chunk #{i} (chars={len(text)}):\n")
            out_lines.append(text + "\n")
            out_lines.append("METADATA: " + json.dumps(meta, ensure_ascii=False) + "\n\n")
            total_chunks += 1

    out_lines.insert(0, f"Total chunks: {total_chunks}\n\n")
    OUTPUT_FILE.write_text("".join(out_lines), encoding="utf-8")
    print(f"Preview escrito en {OUTPUT_FILE} — {total_chunks} chunks")


if __name__ == "__main__":
    main()
