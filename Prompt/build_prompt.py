import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np

from model.model import Recommendation
from Prompt.prompt import SYSTEM_TEMPLATE

BASE = Path(__file__).parent.parent
INDEX_PATH = BASE / "data" / "search_index.pkl"
FAISS_PATH = BASE / "data" / "search_index.faiss"


print("Loading search index...")
with open(INDEX_PATH, "rb") as f:
    _idx = pickle.load(f)

CATALOG: list[dict] = _idx["catalog"]
IDF: dict = _idx["idf"]
MATRIX: np.ndarray = _idx["matrix"]
VOCABULARY: dict = _idx["vocabulary"]
VECTOR_DIMENSION: int = _idx["vector_dimension"]

try:
    import faiss

    if FAISS_PATH.exists():
        FAISS_INDEX = faiss.read_index(str(FAISS_PATH))
    else:
        FAISS_INDEX = faiss.IndexFlatIP(VECTOR_DIMENSION)
        FAISS_INDEX.add(MATRIX.astype("float32"))
except ImportError:
    FAISS_INDEX = None

# Build a URL-to-item lookup for grounding checks.
URL_SET = {item["url"] for item in CATALOG}
NAME_TO_ITEM = {item["name"].lower(): item for item in CATALOG}

backend = "FAISS" if FAISS_INDEX is not None else "NumPy fallback"
print(f"Index loaded: {len(CATALOG)} assessments ({backend})")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z][a-z0-9+#.]*\b", text.lower())


def _query_vector(tokens: list[str]) -> np.ndarray:
    vector = np.zeros(VECTOR_DIMENSION, dtype="float32")
    total = len(tokens) or 1
    freq = Counter(tokens)
    for token, count in freq.items():
        if token in VOCABULARY:
            vector[VOCABULARY[token]] += (count / total) * IDF[token]

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def _tfidf_search(query: str, top_k: int = 15) -> list[dict]:
    """Return top-k catalog items ranked by dense TF-IDF vector similarity."""
    q_toks = _tokenize(query)
    if not q_toks:
        return []

    q_vec = _query_vector(q_toks)
    if FAISS_INDEX is not None:
        scores, indices = FAISS_INDEX.search(q_vec.reshape(1, -1), top_k)
        return [
            CATALOG[i]
            for score, i in zip(scores[0], indices[0])
            if i >= 0 and score > 0
        ]

    scores = MATRIX @ q_vec
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [CATALOG[i] for i in top_indices if scores[i] > 0]


def _format_catalog_snippet(items: list[dict]) -> str:
    """Format catalog items as a compact block for the system prompt."""
    lines = []
    for it in items:
        levels = ",".join(it.get("job_levels", [])[:3]) or "all"
        duration = it.get("duration") or "varies"
        description = it.get("description", "")
        lines.append(
            f'- **{it["name"]}** | type:{it["test_type"]} | '
            f"levels:{levels} | "
            f"duration:{duration} | "
            f'url:{it["url"]}\n  {description[:160]}...'
        )
    return "\n".join(lines)


def _build_system_prompt(user_history_text: str) -> str:
    """Retrieve relevant catalog items and inject into system prompt."""
    lines = [l.strip() for l in user_history_text.split("\n") if l.strip()]
    user_lines = [l for l in lines if l.startswith("USER:")][-3:]
    retrieval_query = " ".join(u.replace("USER:", "").strip() for u in user_lines) or user_history_text

    candidates = _tfidf_search(retrieval_query, top_k=20)
    catalog_text = _format_catalog_snippet(candidates) if candidates else "(no matches found)"
    return SYSTEM_TEMPLATE.format(catalog_section=catalog_text)


_JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            nested = _extract_nested_json(parsed)
            return nested or parsed
    except json.JSONDecodeError:
        pass

    m = _JSON_RE.search(text)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict):
                nested = _extract_nested_json(parsed)
                return nested or parsed
        except json.JSONDecodeError:
            pass

    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(text[start:end])
            if isinstance(parsed, dict):
                nested = _extract_nested_json(parsed)
                return nested or parsed
    except Exception:
        pass
    return None


def _extract_nested_json(parsed: dict) -> dict | None:
    reply = parsed.get("reply")
    if not isinstance(reply, str):
        return None

    nested = _extract_json(reply) if "{" in reply and "}" in reply else None
    if not nested:
        return None

    if "recommendations" in nested or "end_of_conversation" in nested:
        return nested
    return None

def _validate_recommendations(recs: list[dict]) -> list[Recommendation]:
    """
    Hard guard: only allow URLs that actually exist in our catalog.
    Prevents hallucinated recommendations from leaking through.
    """
    valid = []
    for r in recs:
        url = r.get("url", "")
        name = r.get("name", "")
        if url in URL_SET:
            valid.append(Recommendation(
                name=name,
                url=url,
                test_type=r.get("test_type", "K"),
            ))
        else:
            item = NAME_TO_ITEM.get(name.lower())
            if item:
                valid.append(Recommendation(
                    name=item["name"],
                    url=item["url"],
                    test_type=item["test_type"],
                ))
    return valid[:10]
