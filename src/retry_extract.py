"""Retry failed deep extractions with higher max_completion_tokens."""

import json
import base64
import time
from pathlib import Path
from datetime import datetime

from openai import OpenAI

DATA_DIR = Path(__file__).parent.parent / "data"
PIPELINE_DIR = DATA_DIR / "deep_pipeline"
PAPERS_DIR = DATA_DIR / "papers_full"
RESULTS_DIR = DATA_DIR / "deep_results"
MAX_BATCH = 90 * 1024 * 1024

PROMPT = open(Path(__file__).parent / "deep_extract.py").read()
# Extract just the prompt string between the triple quotes
import re
match = re.search(r"EXTRACTION_PROMPT = (\"\"\".*?\"\"\")", PROMPT, re.DOTALL)
EXTRACTION_PROMPT = eval(match.group(1)) if match else ""

def main():
    retry_ids = json.loads(open(PIPELINE_DIR / "retry_ids.json").read())
    print(f"Papers to retry: {len(retry_ids)}")

    # Build batch files
    batch_idx = 0
    current_file = None
    current_size = 0
    total = 0
    batch_files = []

    for cid in sorted(retry_ids):
        pdf_path = PAPERS_DIR / f"{cid}.pdf"
        if not pdf_path.exists():
            continue

        pdf_b64 = base64.standard_b64encode(pdf_path.read_bytes()).decode("utf-8")

        request = {
            "custom_id": cid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-5.4",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXTRACTION_PROMPT},
                        {"type": "file", "file": {
                            "filename": f"{cid}.pdf",
                            "file_data": f"data:application/pdf;base64,{pdf_b64}",
                        }},
                    ],
                }],
                "max_completion_tokens": 32768,
                "response_format": {"type": "json_object"},
            },
        }

        line = json.dumps(request) + "\n"
        line_bytes = len(line.encode("utf-8"))

        if current_file is None or current_size + line_bytes > MAX_BATCH:
            if current_file:
                current_file.close()
                print(f"  {batch_files[-1].name}: {current_size / 1e6:.1f} MB")

            batch_idx += 1
            fname = PIPELINE_DIR / f"retry_{batch_idx:03d}.jsonl"
            batch_files.append(fname)
            current_file = open(fname, "w")
            current_size = 0

        current_file.write(line)
        current_size += line_bytes
        total += 1

    if current_file:
        current_file.close()
        print(f"  {batch_files[-1].name}: {current_size / 1e6:.1f} MB")

    print(f"\nPrepared {len(batch_files)} batch files, {total} papers")

    # Submit
    client = OpenAI()
    tracker_file = PIPELINE_DIR / "retry_tracker.json"
    tracker = {}

    for fpath in batch_files:
        print(f"\n  Uploading {fpath.name}...")
        uploaded = client.files.create(file=open(fpath, "rb"), purpose="batch")
        print(f"  Creating batch...")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        tracker[fpath.name] = {
            "batch_id": batch.id,
            "file_id": uploaded.id,
            "status": batch.status,
        }
        print(f"  Batch {batch.id} ({batch.status})")
        time.sleep(2)

    json.dump(tracker, open(tracker_file, "w"), indent=2)
    print(f"\n{len(tracker)} retry batches submitted.")


if __name__ == "__main__":
    main()
