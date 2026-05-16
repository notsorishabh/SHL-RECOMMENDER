import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
CATALOG_PATH = BASE / "data" / "catalog.json"
INDEX_PATH = BASE / "data" / "search_index.pkl"
FAISS_PATH = BASE / "data" / "search_index.faiss"

KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z][a-z0-9+#.]*\b", text.lower())


def test_type_from_keys(keys: list[str]) -> str:
    codes = [KEY_TO_CODE[key] for key in keys if key in KEY_TO_CODE]
    return ",".join(codes) if codes else "K"


def normalize_item(item: dict) -> dict:
    normalized = dict(item)
    normalized["url"] = normalized.get("url") or normalized.get("link", "")
    normalized["test_type"] = (
        normalized.get("test_type")
        or test_type_from_keys(normalized.get("keys", []))
    )
    normalized.setdefault("job_levels", [])
    normalized.setdefault("languages", [])
    normalized.setdefault("duration", "")
    normalized.setdefault("description", "")
    normalized.setdefault("name", "")
    return normalized


def search_text_for(item: dict) -> str:
    slug_words = item["url"].rstrip("/").split("/")[-1].replace("-", " ")
    return " ".join([
        item["name"],
        item["description"],
        " ".join(item.get("keys", [])),
        " ".join(item.get("job_levels", [])),
        " ".join(item.get("languages", [])),
        item.get("duration", ""),
        slug_words,
    ])


def main() -> None:
    with CATALOG_PATH.open(encoding="utf-8") as f:
        raw_catalog = json.load(f)

    catalog = [normalize_item(item) for item in raw_catalog]
    docs = [tokenize(search_text_for(item)) for item in catalog]

    doc_count = len(docs)
    df = Counter()
    for tokens in docs:
        for token in set(tokens):
            df[token] += 1

    idf = {
        token: math.log((doc_count + 1) / (count + 1)) + 1
        for token, count in df.items()
    }

    vocabulary = {token: i for i, token in enumerate(sorted(idf))}
    vector_dimension = len(vocabulary)
    matrix = np.zeros((len(docs), vector_dimension), dtype="float32")
    for row, tokens in enumerate(docs):
        total = len(tokens) or 1
        freq = Counter(tokens)
        for token, count in freq.items():
            matrix[row, vocabulary[token]] += (count / total) * idf[token]

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)

    faiss_written = False
    try:
        import faiss

        index = faiss.IndexFlatIP(vector_dimension)
        index.add(matrix)
        faiss.write_index(index, str(FAISS_PATH))
        faiss_written = True
    except ImportError:
        if FAISS_PATH.exists():
            FAISS_PATH.unlink()

    with INDEX_PATH.open("wb") as f:
        pickle.dump({
            "catalog": catalog,
            "idf": idf,
            "matrix": matrix,
            "vocabulary": vocabulary,
            "vector_dimension": vector_dimension,
        }, f)

    backend = "FAISS file written" if faiss_written else "pickle only"
    print(
        f"Saved {INDEX_PATH} with {len(catalog)} items, "
        f"{len(idf)} terms, {vector_dimension} dimensions, backend={backend}"
    )


if __name__ == "__main__":
    main()
