import json
from pathlib import Path
import statistics

base = Path(__file__).resolve().parent.parent / "docs_temp"
files = sorted(base.glob("*.jsonl"))

if not files:
    print("No se encontraron archivos JSONL en docs_temp")
    raise SystemExit(1)

summary = []
total_chunks = 0
total_chars = 0
total_words = 0

for f in files:
    lens_chars = []
    lens_words = []
    count = 0
    with f.open("r", encoding="utf-8") as fh:
        for line in fh:
            line=line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"JSON inválido en {f.name}: {e}")
                continue
            text = str(rec.get("chunk_text",""))
            if not text:
                continue
            count += 1
            chars = len(text)
            words = len(text.split())
            lens_chars.append(chars)
            lens_words.append(words)
    if count==0:
        print(f"{f.name}: 0 chunks")
        continue
    total_chunks += count
    total_chars += sum(lens_chars)
    total_words += sum(lens_words)
    summary.append((f.name, count, min(lens_chars), max(lens_chars), int(statistics.mean(lens_chars)), int(statistics.median(lens_chars)), int(statistics.mean(lens_words))))

print("Chunk stats per file:")
print("file,count,min_chars,max_chars,avg_chars,median_chars,avg_words")
for s in summary:
    print(",".join(map(str,s)))

if total_chunks:
    print()
    print(f"TOTAL_CHUNKS={total_chunks}")
    print(f"AVG_CHARS_PER_CHUNK={int(total_chars/total_chunks)}")
    print(f"AVG_WORDS_PER_CHUNK={int(total_words/total_chunks)}")
    # crude token estimate: 0.75 * words (approx)
    est_tokens = int(total_words * 0.75)
    print(f"ESTIMATED_TOTAL_TOKENS={est_tokens}")
else:
    print("No se encontraron chunks válidos.")
