"""
PageIndex Client wrapper — handles document submission and tree retrieval with local caching.
"""

import json
from pathlib import Path
from pageindex import PageIndexClient
from src_pageindex.config import PAGEINDEX_API_KEY, PAGEINDEX_CACHE_DIR


def get_pi_client():
    """Initialize PageIndex client using the unified API key."""
    if not PAGEINDEX_API_KEY:
        raise ValueError("PAGEINDEX_API_KEY is not set in environment or config.")
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _get_cache_path(pdf_name: str) -> Path:
    return PAGEINDEX_CACHE_DIR / f"{pdf_name}.json"


def submit_and_get_tree(pdf_path: Path):
    """
    Submits a PDF to PageIndex, waits for it to be ready, fetches the tree,
    and caches it locally to avoid repeated API calls.
    """
    pdf_name = pdf_path.name
    cache_path = _get_cache_path(pdf_name)

    if cache_path.exists():
        print(f"  [Cache] Loading Document Tree for '{pdf_name}' from local cache...")
        with open(cache_path, "r") as f:
            data = json.load(f)
        return data["tree"]

    print(f"  [API Call] Submitting '{pdf_name}' to PageIndex...")
    client = get_pi_client()
    
    # Needs to be absolute string path
    response = client.submit_document(str(pdf_path.resolve()))
    doc_id = response["document_id"]
    print(f"  [API Call] Document submitted. ID: {doc_id}. Waiting for tree generation...")

    # Poll for completion
    result = client.wait_for_ready(doc_id, timeout=600)
    tree = result["tree"]

    # Cache locally
    print(f"  [Cache] Saving Document Tree for '{pdf_name}' to local cache...")
    with open(cache_path, "w") as f:
        json.dump({"document_id": doc_id, "tree": tree}, f, indent=2)

    return tree
