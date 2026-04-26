"""Query and retrieval logic."""

from .embedder import embed_image, embed_query
from .store import SentryStore


def _search_with_embedding(
    embedding: list[float],
    store: SentryStore,
    n_results: int,
) -> list[dict]:
    hits = store.search(embedding, n_results=n_results)
    results = [
        {
            "source_file": hit["source_file"],
            "start_time": hit["start_time"],
            "end_time": hit["end_time"],
            "similarity_score": hit["score"],
        }
        for hit in hits
    ]
    results.sort(key=lambda r: r["similarity_score"], reverse=True)
    return results


def search_footage(
    query: str,
    store: SentryStore,
    n_results: int = 5,
    verbose: bool = False,
) -> list[dict]:
    """Search indexed footage with a natural language query.

    Args:
        query: Natural language search string.
        store: SentryStore instance to search against.
        n_results: Maximum number of results to return.
        verbose: If True, print debug info to stderr.

    Returns:
        List of result dicts sorted by relevance (best first).
        Each dict contains: source_file, start_time, end_time, similarity_score.
    """
    return _search_with_embedding(
        embed_query(query, verbose=verbose), store, n_results,
    )


def search_footage_by_image(
    image_path: str,
    store: SentryStore,
    n_results: int = 5,
    verbose: bool = False,
) -> list[dict]:
    """Search indexed footage using an image as the query."""
    return _search_with_embedding(
        embed_image(image_path, verbose=verbose), store, n_results,
    )
