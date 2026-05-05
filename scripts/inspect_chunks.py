"""
Adalat-AI Chunk Inspector CLI
==============================
Manually review extraction output for any document.

Usage:
    python scripts/inspect_chunks.py --doc pk-tenancy-punjab-rented-premises-act-2009
    python scripts/inspect_chunks.py --doc uk-employment-equality-act-2010 --limit 20
    python scripts/inspect_chunks.py --doc pk-criminal-code-criminal-procedure-crpc-1898 --search "arrest"
    python scripts/inspect_chunks.py --list
    python scripts/inspect_chunks.py --stats
"""

import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHUNKS_DIR = Path("data/processed/chunks")


def list_available_docs():
    """List all available chunk files."""
    files = sorted(CHUNKS_DIR.glob("*.chunks.json"))
    if not files:
        print("No chunk files found. Run chunker first.")
        return
    print(f"\n{'#':<4} {'Document':<60} {'Chunks'}")
    print("-" * 80)
    for i, f in enumerate(files, 1):
        chunks = json.loads(f.read_text(encoding="utf-8"))
        print(f"{i:<4} {f.stem.replace('.chunks', ''):<60} {len(chunks)}")


def load_chunks(doc_name: str) -> list:
    """Load chunks for a document."""
    pattern = f"{doc_name}*.chunks.json"
    files = list(CHUNKS_DIR.glob(pattern))
    if not files:
        print(f"No chunks found for: {doc_name}")
        print(f"Available: {[f.stem for f in CHUNKS_DIR.glob('*.chunks.json')]}")
        sys.exit(1)
    return json.loads(files[0].read_text(encoding="utf-8"))


def show_stats(chunks: list, doc_name: str):
    """Show statistics for a document's chunks."""
    tokens = [c["token_count"] for c in chunks]
    hierarchies = [str(c["hierarchy"]) for c in chunks]
    categories = defaultdict(int)
    for c in chunks:
        categories[c.get("category", "unknown")] += 1

    print(f"\n{'='*60}")
    print(f"STATS: {doc_name}")
    print(f"{'='*60}")
    print(f"Total chunks    : {len(chunks)}")
    print(f"Avg tokens      : {sum(tokens)/len(tokens):.0f}")
    print(f"Min tokens      : {min(tokens)}")
    print(f"Max tokens      : {max(tokens)}")
    print(f"Chunks < 50 tok : {sum(1 for t in tokens if t < 50)}")
    print(f"Chunks > 700 tok: {sum(1 for t in tokens if t > 700)}")
    print(f"\nHierarchy levels found:")
    level_counts = defaultdict(int)
    for c in chunks:
        for level in c.get("hierarchy", {}).keys():
            level_counts[level] += 1
    for level, count in sorted(level_counts.items()):
        print(f"  {level:<15}: {count}")

    print(f"\nCross-references: "
          f"{sum(len(c.get('cross_references',[])) for c in chunks)} total")

    # Token distribution
    print(f"\nToken distribution:")
    ranges = [(0,100),(100,200),(200,400),(400,600),(600,800),(800,9999)]
    for lo, hi in ranges:
        count = sum(1 for t in tokens if lo <= t < hi)
        bar = "█" * (count // 2)
        print(f"  {lo:>4}-{hi:<5}: {count:>4} {bar}")


def show_chunks(chunks: list, limit: int = 10,
                search: str = None, offset: int = 0):
    """Display chunks with full detail."""
    if search:
        filtered = [c for c in chunks
                    if search.lower() in c["text"].lower() or
                    search.lower() in c["breadcrumb"].lower()]
        print(f"\nFound {len(filtered)} chunks matching '{search}'")
        chunks = filtered

    display = chunks[offset:offset + limit]

    for i, chunk in enumerate(display, offset + 1):
        print(f"\n{'─'*60}")
        print(f"[{i}/{len(chunks)}] {chunk['chunk_id']}")
        print(f"{'─'*60}")
        print(f"Breadcrumb : {chunk['breadcrumb']}")
        print(f"Tokens     : {chunk['token_count']}")
        print(f"Pages      : {chunk['page_start']}–{chunk['page_end']}")
        print(f"Hierarchy  : {chunk['hierarchy']}")
        if chunk.get("cross_references"):
            print(f"Cross-refs : {chunk['cross_references'][:5]}")
        if chunk.get("currency_warning"):
            print(f"⚠️  Warning : {chunk['currency_warning']}")
        if chunk.get("requires_escalation_cue"):
            print(f"⚠️  Escalation cue required")
        print(f"\nText:")
        print(f"  {chunk['text'][:500].replace(chr(10), chr(10)+'  ')}")

    if len(chunks) > offset + limit:
        remaining = len(chunks) - offset - limit
        print(f"\n... {remaining} more chunks. "
              f"Use --offset {offset+limit} to see next batch.")


def interactive_mode(doc_name: str):
    """Simple interactive browser."""
    chunks = load_chunks(doc_name)
    show_stats(chunks, doc_name)

    offset = 0
    limit = 5

    while True:
        print(f"\n{'='*60}")
        print(f"Commands: [n]ext  [p]rev  [s]earch  [g]oto  [q]uit")
        print(f"Showing {offset+1}-{min(offset+limit, len(chunks))} "
              f"of {len(chunks)} chunks")
        print(f"{'='*60}")

        show_chunks(chunks, limit=limit, offset=offset)

        cmd = input("\n> ").strip().lower()

        if cmd == "q" or cmd == "quit":
            break
        elif cmd == "n" or cmd == "":
            offset = min(offset + limit, len(chunks) - limit)
        elif cmd == "p":
            offset = max(0, offset - limit)
        elif cmd.startswith("s "):
            search_term = cmd[2:]
            show_chunks(chunks, limit=20, search=search_term)
            input("\nPress Enter to continue...")
        elif cmd.startswith("g "):
            try:
                offset = max(0, int(cmd[2:]) - 1)
            except ValueError:
                print("Invalid number")
        else:
            print("Unknown command")


def main():
    parser = argparse.ArgumentParser(
        description="Adalat-AI Chunk Inspector"
    )
    parser.add_argument("--doc", help="Document name to inspect")
    parser.add_argument("--list", action="store_true",
                        help="List available documents")
    parser.add_argument("--stats", action="store_true",
                        help="Show stats for document")
    parser.add_argument("--search", help="Search for text in chunks")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of chunks to show (default: 5)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Start from chunk number")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive browsing mode")

    args = parser.parse_args()

    if args.list:
        list_available_docs()
        return

    if not args.doc:
        print("Usage: python scripts/inspect_chunks.py --doc <document_name>")
        print("       python scripts/inspect_chunks.py --list")
        parser.print_help()
        return

    chunks = load_chunks(args.doc)

    if args.interactive:
        interactive_mode(args.doc)
        return

    if args.stats or not args.search:
        show_stats(chunks, args.doc)

    if args.search:
        show_chunks(chunks, limit=args.limit,
                    search=args.search, offset=args.offset)
    elif not args.stats:
        show_chunks(chunks, limit=args.limit, offset=args.offset)


if __name__ == "__main__":
    main()