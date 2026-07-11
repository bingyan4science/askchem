"""
AskChem Extraction Quality Evaluation.

Measures precision, recall, and field-level accuracy of LLM-extracted claims
against a human-annotated gold standard.

Gold standard format (JSON per paper):
{
    "doi": "10.1234/...",
    "source": "abstract" | "full_paper",
    "annotator": "name",
    "claims": [
        {
            "claim_type": "reaction",
            "verbatim_quote": "...",
            "fields": {"reaction_type": "Suzuki coupling", "yield_percent": 95, ...}
        },
        ...
    ]
}

Usage:
    python src/evaluate_extraction.py evaluate data/gold_standard/
    python src/evaluate_extraction.py report data/gold_standard/
    python src/evaluate_extraction.py create-template --doi 10.1234/example
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from difflib import SequenceMatcher

sys.path.insert(0, str(Path(__file__).parent))


def load_gold_standard(gold_dir: Path) -> list[dict]:
    """Load all gold-standard annotation files."""
    papers = []
    for f in sorted(gold_dir.glob("*.json")):
        with open(f) as fh:
            papers.append(json.load(fh))
    return papers


def load_extracted_claims(doi: str, db_path: Path = None) -> list[dict]:
    """Load extracted claims for a paper from the database."""
    if db_path:
        import os
        os.environ["CHEMTREE_DB"] = str(db_path)

    from askchem import db as cdb
    return cdb.get_claims_by_doi(doi)


def match_claims(gold_claims: list[dict], extracted_claims: list[dict],
                 quote_threshold: float = 0.5) -> dict:
    """
    Match gold-standard claims to extracted claims using quote similarity.

    Returns:
        {
            "matches": [(gold_idx, extracted_idx, similarity)],
            "unmatched_gold": [gold_idx, ...],
            "unmatched_extracted": [extracted_idx, ...],
        }
    """
    matches = []
    used_extracted = set()
    used_gold = set()

    similarity_matrix = []
    for gi, gc in enumerate(gold_claims):
        for ei, ec in enumerate(extracted_claims):
            gq = gc.get("verbatim_quote", "").lower().strip()
            eq = ec.get("verbatim_quote", "").lower().strip()
            if not gq or not eq:
                sim = 0.0
            else:
                sim = SequenceMatcher(None, gq, eq).ratio()

            gt = gc.get("claim_type", "")
            et = ec.get("claim_type", "")
            type_match = 1.0 if gt == et else 0.5

            score = sim * 0.7 + type_match * 0.3
            similarity_matrix.append((score, gi, ei, sim))

    similarity_matrix.sort(reverse=True)

    for score, gi, ei, sim in similarity_matrix:
        if gi in used_gold or ei in used_extracted:
            continue
        if sim >= quote_threshold:
            matches.append((gi, ei, sim))
            used_gold.add(gi)
            used_extracted.add(ei)

    unmatched_gold = [i for i in range(len(gold_claims)) if i not in used_gold]
    unmatched_extracted = [i for i in range(len(extracted_claims)) if i not in used_extracted]

    return {
        "matches": matches,
        "unmatched_gold": unmatched_gold,
        "unmatched_extracted": unmatched_extracted,
    }


def evaluate_field_accuracy(gold_claim: dict, extracted_claim: dict) -> dict:
    """Compare field-level accuracy between a matched pair."""
    gold_fields = gold_claim.get("fields", {})
    results = {}

    for field_name, gold_value in gold_fields.items():
        ext_value = extracted_claim.get(field_name)

        if ext_value is None or ext_value == "" or ext_value == []:
            results[field_name] = {"status": "missing", "gold": gold_value, "extracted": None}
        elif isinstance(gold_value, (int, float)) and isinstance(ext_value, (int, float, str)):
            try:
                ext_num = float(str(ext_value).strip().rstrip("%"))
                gold_num = float(gold_value)
                if abs(ext_num - gold_num) < 0.01 * max(abs(gold_num), 1):
                    results[field_name] = {"status": "correct", "gold": gold_value, "extracted": ext_value}
                else:
                    results[field_name] = {"status": "incorrect", "gold": gold_value, "extracted": ext_value}
            except (ValueError, TypeError):
                results[field_name] = {"status": "incorrect", "gold": gold_value, "extracted": ext_value}
        elif isinstance(gold_value, str):
            sim = SequenceMatcher(None, str(gold_value).lower(), str(ext_value).lower()).ratio()
            status = "correct" if sim > 0.8 else "partial" if sim > 0.5 else "incorrect"
            results[field_name] = {"status": status, "gold": gold_value, "extracted": ext_value}
        else:
            results[field_name] = {"status": "unchecked", "gold": gold_value, "extracted": ext_value}

    return results


def evaluate_paper(gold_paper: dict, db_path: Path = None) -> dict:
    """Evaluate extraction quality for a single paper."""
    doi = gold_paper["doi"]
    gold_claims = gold_paper["claims"]
    extracted_claims = load_extracted_claims(doi, db_path)

    matching = match_claims(gold_claims, extracted_claims)

    tp = len(matching["matches"])
    fn = len(matching["unmatched_gold"])
    fp = len(matching["unmatched_extracted"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    field_results = []
    type_accuracy = Counter()
    for gi, ei, sim in matching["matches"]:
        gc = gold_claims[gi]
        ec = extracted_claims[ei]

        if gc.get("claim_type") == ec.get("claim_type"):
            type_accuracy["correct"] += 1
        else:
            type_accuracy["incorrect"] += 1

        fr = evaluate_field_accuracy(gc, ec)
        field_results.append({"gold_idx": gi, "extracted_idx": ei, "similarity": sim, "fields": fr})

    return {
        "doi": doi,
        "source": gold_paper.get("source", "unknown"),
        "gold_count": len(gold_claims),
        "extracted_count": len(extracted_claims),
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "type_accuracy": dict(type_accuracy),
        "field_results": field_results,
    }


def evaluate_all(gold_dir: Path, db_path: Path = None) -> dict:
    """Evaluate extraction quality across all gold-standard papers."""
    papers = load_gold_standard(gold_dir)
    if not papers:
        print(f"No gold-standard files found in {gold_dir}")
        return {}

    results = []
    for paper in papers:
        print(f"  Evaluating {paper['doi']}...", flush=True)
        r = evaluate_paper(paper, db_path)
        results.append(r)

    total_tp = sum(r["true_positives"] for r in results)
    total_fn = sum(r["false_negatives"] for r in results)
    total_fp = sum(r["false_positives"] for r in results)

    macro_precision = sum(r["precision"] for r in results) / len(results)
    macro_recall = sum(r["recall"] for r in results) / len(results)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    field_stats = defaultdict(Counter)
    for r in results:
        for fr in r["field_results"]:
            for field_name, info in fr["fields"].items():
                field_stats[field_name][info["status"]] += 1

    abstract_results = [r for r in results if r["source"] == "abstract"]
    fullpaper_results = [r for r in results if r["source"] == "full_paper"]

    summary = {
        "total_papers": len(results),
        "abstract_papers": len(abstract_results),
        "full_paper_papers": len(fullpaper_results),
        "micro_precision": round(micro_precision, 3),
        "micro_recall": round(micro_recall, 3),
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),
        "total_gold_claims": sum(r["gold_count"] for r in results),
        "total_extracted_claims": sum(r["extracted_count"] for r in results),
        "field_accuracy": {k: dict(v) for k, v in field_stats.items()},
        "per_paper": results,
    }

    if abstract_results:
        summary["abstract_precision"] = round(
            sum(r["precision"] for r in abstract_results) / len(abstract_results), 3)
        summary["abstract_recall"] = round(
            sum(r["recall"] for r in abstract_results) / len(abstract_results), 3)

    if fullpaper_results:
        summary["full_paper_precision"] = round(
            sum(r["precision"] for r in fullpaper_results) / len(fullpaper_results), 3)
        summary["full_paper_recall"] = round(
            sum(r["recall"] for r in fullpaper_results) / len(fullpaper_results), 3)

    return summary


def create_template(doi: str, output_dir: Path):
    """Create a gold-standard annotation template for a paper."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_doi = doi.replace("/", "_").replace(".", "-")
    template = {
        "doi": doi,
        "source": "abstract",
        "annotator": "",
        "annotation_date": "",
        "claims": [
            {
                "claim_type": "reaction",
                "verbatim_quote": "PASTE EXACT TEXT FROM PAPER",
                "fields": {
                    "reaction_type": "",
                    "yield_percent": None,
                }
            }
        ],
        "notes": "Delete the example claim above and add real annotations."
    }
    out_path = output_dir / f"{safe_doi}.json"
    with open(out_path, "w") as f:
        json.dump(template, f, indent=2)
    print(f"Template created: {out_path}")


def print_report(summary: dict):
    """Print a human-readable evaluation report."""
    print("\n" + "=" * 60)
    print("AskChem Extraction Quality Report")
    print("=" * 60)
    print(f"\nPapers evaluated: {summary['total_papers']}")
    print(f"  Abstract-only: {summary['abstract_papers']}")
    print(f"  Full-paper:    {summary['full_paper_papers']}")
    print(f"\nGold-standard claims: {summary['total_gold_claims']}")
    print(f"Extracted claims:     {summary['total_extracted_claims']}")
    print(f"\n{'Metric':<25} {'Micro':>8} {'Macro':>8}")
    print("-" * 45)
    print(f"{'Precision':<25} {summary['micro_precision']:>8.1%} {summary['macro_precision']:>8.1%}")
    print(f"{'Recall':<25} {summary['micro_recall']:>8.1%} {summary['macro_recall']:>8.1%}")

    if "abstract_precision" in summary:
        print(f"\n{'Abstract precision':<25} {summary['abstract_precision']:>8.1%}")
        print(f"{'Abstract recall':<25} {summary['abstract_recall']:>8.1%}")
    if "full_paper_precision" in summary:
        print(f"{'Full-paper precision':<25} {summary['full_paper_precision']:>8.1%}")
        print(f"{'Full-paper recall':<25} {summary['full_paper_recall']:>8.1%}")

    if summary.get("field_accuracy"):
        print(f"\n{'Field':<25} {'Correct':>8} {'Partial':>8} {'Incorrect':>8} {'Missing':>8}")
        print("-" * 65)
        for field_name, counts in sorted(summary["field_accuracy"].items()):
            total = sum(counts.values())
            print(f"{field_name:<25} "
                  f"{counts.get('correct', 0):>8} "
                  f"{counts.get('partial', 0):>8} "
                  f"{counts.get('incorrect', 0):>8} "
                  f"{counts.get('missing', 0):>8}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AskChem extraction quality evaluation")
    sub = parser.add_subparsers(dest="command")

    eval_p = sub.add_parser("evaluate", help="Run evaluation against gold standard")
    eval_p.add_argument("gold_dir", type=Path, help="Directory with gold-standard JSON files")
    eval_p.add_argument("--db", type=Path, help="Path to chemtree.db")
    eval_p.add_argument("--output", type=Path, help="Save results to JSON file")

    report_p = sub.add_parser("report", help="Print evaluation report")
    report_p.add_argument("gold_dir", type=Path, help="Directory with gold-standard JSON files")
    report_p.add_argument("--db", type=Path, help="Path to chemtree.db")

    tmpl_p = sub.add_parser("create-template", help="Create annotation template")
    tmpl_p.add_argument("--doi", required=True, help="Paper DOI")
    tmpl_p.add_argument("--output-dir", type=Path, default=Path("data/gold_standard"))

    args = parser.parse_args()

    if args.command == "evaluate":
        summary = evaluate_all(args.gold_dir, args.db)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(summary, indent=2))

    elif args.command == "report":
        summary = evaluate_all(args.gold_dir, args.db)
        print_report(summary)

    elif args.command == "create-template":
        create_template(args.doi, args.output_dir)

    else:
        parser.print_help()
