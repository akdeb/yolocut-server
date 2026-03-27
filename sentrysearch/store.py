"""ChromaDB vector store."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import chromadb


DEFAULT_DB_PATH = Path.home() / ".sentrysearch" / "db"

# Backend → collection name mapping
_COLLECTION_NAMES = {
    "gemini": "dashcam_chunks",
    "local": "dashcam_chunks_local",
}


class BackendMismatchError(RuntimeError):
    """Raised when search backend doesn't match the indexed backend."""


def detect_backend(db_path: str | Path | None = None) -> str | None:
    """Return the backend that has indexed data, or None if empty.

    If both backends have data, returns 'gemini' (the default).
    """
    db_path = str(db_path or DEFAULT_DB_PATH)
    if not Path(db_path).exists():
        return None
    client = chromadb.PersistentClient(path=db_path)
    existing = {c.name for c in client.list_collections()}
    # Check gemini first (default/legacy)
    if "dashcam_chunks" in existing:
        col = client.get_collection("dashcam_chunks")
        if col.count() > 0:
            return "gemini"
    if "dashcam_chunks_local" in existing:
        col = client.get_collection("dashcam_chunks_local")
        if col.count() > 0:
            return "local"
    return None


def _make_chunk_id(source_file: str, start_time: float) -> str:
    """Deterministic chunk ID from source file + start time."""
    raw = f"{source_file}:{start_time}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class SentryStore:
    """Persistent vector store backed by ChromaDB."""

    def __init__(self, db_path: str | Path | None = None, backend: str = "gemini"):
        db_path = str(db_path or DEFAULT_DB_PATH)
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=db_path)
        self._backend = backend
        # Separate collection per backend so vectors never mix.
        # Legacy collection "dashcam_chunks" (no suffix) is treated as gemini.
        collection_name = "dashcam_chunks" if backend == "gemini" else f"dashcam_chunks_{backend}"
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding_backend": backend},
        )

    @property
    def collection(self) -> chromadb.Collection:
        return self._collection

    def get_backend(self) -> str:
        """Return the backend this index was built with."""
        meta = self._collection.metadata or {}
        return meta.get("embedding_backend", "gemini")

    def check_backend(self, backend: str) -> None:
        """Raise BackendMismatchError if *backend* doesn't match the index."""
        indexed_backend = self.get_backend()
        if indexed_backend != backend:
            raise BackendMismatchError(
                f"This index was built with the {indexed_backend} backend. "
                f"Search with --backend {indexed_backend} or re-index with "
                f"--backend {backend}."
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunk(
        self,
        chunk_id: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        """Store a single chunk embedding with metadata.

        Required metadata keys: source_file, start_time, end_time.
        An indexed_at ISO timestamp is added automatically.
        """
        meta = {
            "source_file": metadata["source_file"],
            "start_time": float(metadata["start_time"]),
            "end_time": float(metadata["end_time"]),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        # Carry over any extra metadata the caller provides
        for key in metadata:
            if key not in meta and key != "embedding":
                meta[key] = metadata[key]

        self._collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            metadatas=[meta],
        )

    def add_chunks(self, chunks: list[dict]) -> None:
        """Batch-store chunks. Each dict must have 'embedding' and metadata keys."""
        now = datetime.now(timezone.utc).isoformat()
        ids = []
        embeddings = []
        metadatas = []

        for chunk in chunks:
            chunk_id = _make_chunk_id(chunk["source_file"], chunk["start_time"])
            ids.append(chunk_id)
            embeddings.append(chunk["embedding"])
            metadatas.append({
                "source_file": chunk["source_file"],
                "start_time": float(chunk["start_time"]),
                "end_time": float(chunk["end_time"]),
                "indexed_at": now,
            })

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[dict]:
        """Return top N results with distances and metadata."""
        count = self._collection.count()
        if count == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
        )

        hits = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            hits.append({
                "source_file": meta["source_file"],
                "start_time": meta["start_time"],
                "end_time": meta["end_time"],
                "score": 1.0 - distance,  # cosine distance → similarity
                "distance": distance,
            })
        return hits

    def is_indexed(self, source_file: str) -> bool:
        """Check whether any chunks from source_file are already stored."""
        results = self._collection.get(
            where={"source_file": source_file},
            limit=1,
        )
        return len(results["ids"]) > 0

    def get_stats(self) -> dict:
        """Return store statistics."""
        total = self._collection.count()
        if total == 0:
            return {"total_chunks": 0, "unique_source_files": 0, "source_files": []}

        # Fetch all metadata (only the fields we need)
        all_meta = self._collection.get(include=["metadatas"])
        source_files = sorted({m["source_file"] for m in all_meta["metadatas"]})
        return {
            "total_chunks": total,
            "unique_source_files": len(source_files),
            "source_files": source_files,
        }
