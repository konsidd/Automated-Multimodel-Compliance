"""
scripts/index_policies.py
──────────────────────────
Index all .txt policy documents from data/policies/ into the FAISS vector store.

Usage:
  python scripts/index_policies.py
  python scripts/index_policies.py --policies-dir path/to/my/policies
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.chunking import chunk_document
from app.retrieval.retriever import get_vector_store
from app.telemetry.logging_config import get_logger, setup_logging
from app.utils.config import get_config

setup_logging()
logger = get_logger("index_policies")


def index_policies(policies_dir: str | None = None) -> None:
    cfg = get_config()
    root = cfg.root_dir
    dir_path = Path(policies_dir) if policies_dir else (root / cfg.compliance.policies_dir)

    if not dir_path.exists():
        logger.error("policies_dir_not_found", path=str(dir_path))
        sys.exit(1)

    store = get_vector_store()
    files = list(dir_path.glob("**/*.txt")) + list(dir_path.glob("**/*.md"))

    if not files:
        logger.warning("no_policy_files_found", dir=str(dir_path))
        return

    total_chunks = 0
    for fpath in files:
        text = fpath.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_document(text, source=str(fpath))
        store.add_chunks(chunks)
        total_chunks += len(chunks)
        logger.info("file_indexed", path=str(fpath), chunks=len(chunks))

    store.save()
    logger.info("indexing_complete", files=len(files), total_chunks=total_chunks)
    print(f"\n✅  Indexed {len(files)} file(s) → {total_chunks} chunks → {store.total_vectors} vectors in FAISS.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index policy documents into FAISS vector store")
    parser.add_argument("--policies-dir", type=str, default=None, help="Override policies directory")
    args = parser.parse_args()
    index_policies(args.policies_dir)
