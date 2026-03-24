#!/usr/bin/env python3
"""Validate and clean JSONL files under docs/ per project rules.

Checks and fixes applied:
- parse each line as JSON
- ensure required metadata fields exist
- ensure chunk_text non-empty and <= max tokens (heuristic)
- ensure chunk_id unique across files (fix by appending suffix)
- normalize date_range to allowed formats
- enforce sensitivity policy (contacts, CI/CD/secrets -> medium)
- set default visibility to 'public' if missing

Backups originals to docs/jsonl_backups_<timestamp>/

Usage: python scripts/validate_and_clean_jsonl.py
"""
import json
import os
import re
from pathlib import Path
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
BACKUP_DIR = DOCS / f"jsonl_backups_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_FIELDS = ["chunk_id", "file_name", "doc_type", "chunk_text", "language"]

# approximate max tokens per doc_type
MAX_TOKENS_BY_TYPE = {
    "biography": 200,
    "cv": 300,
    "skills": 200,
    "readme": 300,
    "summary": 120,
}
DEFAULT_MAX_TOKENS = 300

DATE_RANGE_RE = re.compile(r"^(\d{4}-\d{2})(/(\d{4}-\d{2}))?$")

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{6,}\d")
CI_KEYWORDS = re.compile(r"\b(ci/cd|ci cd|secret|secrets|credential|credentials|api[_\- ]?key|token)\b", re.I)


def est_tokens(text: str) -> int:
    # crude approximation: word count
    return len(text.split())


def normalize_date_range(value):
    if not value:
        return "unknown"
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("actual", "unknown"):
            return v
        # accept formats like YYYY-MM or YYYY-MM/YYYY-MM
        if DATE_RANGE_RE.match(value):
            return value
    return "unknown"


def detect_sensitive(text: str, metadata: dict) -> str:
    # if metadata already marks medium, keep if valid
    sens = metadata.get("sensitivity", "").lower()
    if sens in ("medium", "low"):
        return sens
    # detect emails, phones or CI/CD keywords
    if EMAIL_RE.search(text) or PHONE_RE.search(text) or CI_KEYWORDS.search(text):
        return "medium"
    return "low"


def normalize_visibility(value):
    if not value:
        return "public"
    v = value.lower()
    if v in ("public", "private", "internal"):
        return v
    return "public"


def main():
    jsonl_files = list(DOCS.rglob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found under docs/")
        return

    all_chunk_ids = set()
    duplicates = 0
    fixed_ids = 0
    fixed_date = 0
    fixed_sensitivity = 0
    fixed_visibility = 0
    total_lines = 0

    for f in jsonl_files:
        rel = f.relative_to(ROOT)
        print(f"Processing {rel}")
        # backup
        backup_path = BACKUP_DIR / f.name
        backup_path.write_bytes(f.read_bytes())

        out_lines = []
        seen_in_file = set()

        with f.open("r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(f"  Line {idx}: JSON parse error: {e} — skipping")
                    continue

                # ensure required fields
                for key in REQUIRED_FIELDS:
                    if key not in obj:
                        # fill sensible default
                        if key == "chunk_id":
                            base = f"{f.stem}_{idx:04d}"
                            obj["chunk_id"] = base
                        elif key == "file_name":
                            obj["file_name"] = f.name
                        elif key == "doc_type":
                            obj["doc_type"] = obj.get("doc_type", "readme")
                        elif key == "chunk_text":
                            obj["chunk_text"] = obj.get("text", "")
                        elif key == "language":
                            obj["language"] = obj.get("language", "es")

                # chunk_text non-empty
                text = obj.get("chunk_text", "")
                if not isinstance(text, str) or not text.strip():
                    print(f"  Line {idx}: empty chunk_text — skipping")
                    continue

                # date_range normalize
                if "date_range" in obj:
                    norm = normalize_date_range(obj.get("date_range"))
                    if norm != obj.get("date_range"):
                        obj["date_range"] = norm
                        fixed_date += 1
                else:
                    obj["date_range"] = "unknown"

                # sensitivity policy
                sens = detect_sensitive(text, obj)
                if obj.get("sensitivity", "").lower() != sens:
                    obj["sensitivity"] = sens
                    fixed_sensitivity += 1

                # visibility default
                vis = normalize_visibility(obj.get("visibility"))
                if obj.get("visibility") != vis:
                    obj["visibility"] = vis
                    fixed_visibility += 1

                # tokens check
                doc_type = obj.get("doc_type", "readme")
                max_tokens = MAX_TOKENS_BY_TYPE.get(doc_type, DEFAULT_MAX_TOKENS)
                tok_est = est_tokens(text)
                obj["chunk_tokens_est"] = tok_est
                if tok_est > max_tokens:
                    print(f"  Line {idx}: token estimate {tok_est} > max {max_tokens} (doc_type={doc_type})")
                    # do not auto-split here; flag and keep the chunk but user may re-chunk later

                # ensure chunk_id uniqueness
                cid = obj.get("chunk_id")
                if cid in all_chunk_ids:
                    # make unique by appending counter
                    duplicates += 1
                    suffix = 1
                    new_cid = f"{cid}_{suffix}"
                    while new_cid in all_chunk_ids or new_cid in seen_in_file:
                        suffix += 1
                        new_cid = f"{cid}_{suffix}"
                    obj["chunk_id"] = new_cid
                    fixed_ids += 1
                    cid = new_cid

                all_chunk_ids.add(cid)
                seen_in_file.add(cid)

                out_lines.append(json.dumps(obj, ensure_ascii=False))

        # overwrite the file with cleaned content
        with f.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(out_lines) + ("\n" if out_lines else ""))

    print("\nSummary:")
    print(f"  JSONL files processed: {len(jsonl_files)}")
    print(f"  Total chunks examined: {total_lines}")
    print(f"  Duplicates fixed: {fixed_ids}")
    print(f"  Date fields normalized: {fixed_date}")
    print(f"  Sensitivity fields set/fixed: {fixed_sensitivity}")
    print(f"  Visibility fields set/fixed: {fixed_visibility}")
    print(f"  Backups saved to: {BACKUP_DIR}")


if __name__ == "__main__":
    main()
