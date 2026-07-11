#!/usr/bin/env python3
"""Push Gemini-verified contradictions to prod.

Two modes:

  * ``--seed deploy/contradictions_seed_v1.json``   (recommended)
      Regenerate the in-repo seed JSON from a raw ``gemini_verified_*.json``
      and (optionally) push it to prod. The seed is what
      ``deploy_to_vps.sh`` re-installs on every deploy, so writing the
      seed is the durable path.

  * ``--push`` (default if neither --seed nor --push given to keep the
      legacy behaviour)
      One-shot direct push to prod: filter to confirmed/rejected,
      DELETE+INSERT the prod ``contradictions`` table, restart askchem.

Usage::

    # Regenerate the deploy seed (the 556 confirmed) from the raw file:
    python3 scripts/upload_contradictions.py \
        data/gemini_verified_viewfree.checkpoint.json \
        --seed deploy/contradictions_seed_v1.json

    # Also push immediately (skip waiting for the next deploy):
    python3 scripts/upload_contradictions.py \
        data/gemini_verified_viewfree.checkpoint.json \
        --seed deploy/contradictions_seed_v1.json --push

The earlier inline ``ssh "$VPS" "python3 -c <repr(script)>"`` path is
gone — nested-quote escaping broke on the prod box. We now stream the
insert script over stdin via ``ssh ... 'python3 -' < script``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SERVER = "root@YOUR_VPS_HOST"
DB_PATH = "/opt/askchem/chemtree.db"


REMOTE_INSERT = r"""
import json, sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
conn.execute("PRAGMA journal_mode=WAL")
records = json.loads(open(sys.argv[2]).read())
conn.execute("DELETE FROM contradictions")
inserted = 0
for r in records:
    try:
        conn.execute(
            "INSERT INTO contradictions "
            "(claim_id_1, claim_id_2, view_id, node_path, "
            "paw_verdict, gemini_verdict, gemini_explanation, "
            "confidence, detected_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (r["claim_id_1"], r["claim_id_2"], r["view_id"], r["node_path"],
             r["paw_verdict"], r["gemini_verdict"],
             r["gemini_explanation"], r["confidence"], r["detected_at"]))
        inserted += 1
    except Exception as e:
        print(f"err: {e}", file=sys.stderr)
conn.commit()
total = conn.execute("SELECT COUNT(*) FROM contradictions").fetchone()[0]
confirmed = conn.execute(
    "SELECT COUNT(*) FROM contradictions "
    "WHERE gemini_verdict = 'confirmed'").fetchone()[0]
print(f"inserted={inserted}  total={total}  confirmed={confirmed}")
"""


def build_records(verified: list[dict], confirmed_only: bool) -> list[dict]:
    now = datetime.now(timezone.utc).isoformat()
    pool = [
        v for v in verified
        if v.get("gemini_verdict") in (
            {"confirmed"} if confirmed_only else {"confirmed", "rejected"}
        )
    ]
    records = []
    for r in pool:
        ids = sorted([r["claim_id_1"], r["claim_id_2"]])
        records.append({
            "claim_id_1": ids[0],
            "claim_id_2": ids[1],
            "view_id": r.get("view_id") or "all",
            "node_path": r.get("node_path") or (r.get("subject") or "")[:200],
            "paw_verdict": r.get("paw_verdict") or "none",
            "gemini_verdict": r.get("gemini_verdict"),
            "gemini_explanation": r.get("gemini_explanation") or "",
            "confidence": float(r.get("confidence") or 0.0),
            "detected_at": now,
        })
    return records


def push_to_prod(records: list[dict]) -> None:
    tmp_local = Path("/tmp/contradictions_upload.json")
    tmp_local.write_text(json.dumps(records))
    print(f"Uploading {len(records):,} records to {SERVER}…", flush=True)
    subprocess.run(
        ["scp", str(tmp_local), f"{SERVER}:/tmp/contradictions_upload.json"],
        check=True,
    )
    # Pipe the insert script over stdin instead of `-c <repr>` so bash
    # never sees the embedded quotes. Pass DB + JSON paths as argv.
    result = subprocess.run(
        ["ssh", SERVER,
         f"/opt/askchem/venv/bin/python3 - {DB_PATH} "
         f"/tmp/contradictions_upload.json"],
        input=REMOTE_INSERT, capture_output=True, text=True,
    )
    print(result.stdout, end="")
    if result.stderr:
        print(f"STDERR: {result.stderr}", file=sys.stderr)
    if result.returncode != 0:
        raise SystemExit(f"insert step failed (rc={result.returncode})")
    subprocess.run(["ssh", SERVER, "systemctl restart askchem"], check=True)
    print("askchem restarted.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "input_file",
        help="Raw gemini_verified_*.json input (list of verdicts).",
    )
    ap.add_argument(
        "--seed", metavar="PATH",
        help="Also write a slim deploy-seed JSON (confirmed-only) to PATH. "
             "This is what deploy_to_vps.sh re-installs on every deploy.",
    )
    ap.add_argument(
        "--push", action="store_true",
        help="Push the FULL records (confirmed + rejected) to prod now. "
             "Without --push, the script only regenerates the seed.",
    )
    args = ap.parse_args()

    verified = json.loads(Path(args.input_file).read_text())
    counts = {"confirmed": 0, "rejected": 0, "error": 0}
    for v in verified:
        counts[v.get("gemini_verdict", "error") or "error"] = \
            counts.get(v.get("gemini_verdict") or "error", 0) + 1
    print(f"input: {args.input_file}  total={len(verified):,}  "
          f"confirmed={counts.get('confirmed', 0):,}  "
          f"rejected={counts.get('rejected', 0):,}  "
          f"errors={counts.get('error', 0):,}")

    if args.seed:
        seed_records = build_records(verified, confirmed_only=True)
        # Strip detected_at — the seed is timeless; deploy-time inserts
        # stamp it with the current deploy time.
        for r in seed_records:
            r.pop("detected_at", None)
        Path(args.seed).parent.mkdir(parents=True, exist_ok=True)
        Path(args.seed).write_text(
            json.dumps(seed_records, ensure_ascii=False, indent=2)
        )
        print(f"wrote seed: {args.seed}  ({len(seed_records):,} confirmed)")

    if args.push:
        push_records = build_records(verified, confirmed_only=False)
        if not push_records:
            print("nothing to push (no confirmed/rejected records).")
            return 0
        push_to_prod(push_records)

    if not args.seed and not args.push:
        print(
            "no action taken. Pass --seed PATH to (re)generate the deploy "
            "seed, --push to upload to prod, or both.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
