from __future__ import annotations
import logging
import re
from typing import Optional, Dict, Any, List
from pypdf import PdfReader

from .llm_client import OpenAICompatibleClient

# Grammar is optional; if you don't use it, keep None.
try:
    from .llm_grammar import GRAMMAR_JSON_INT_OR_NULL
except Exception:
    GRAMMAR_JSON_INT_OR_NULL = None  # type: ignore

log = logging.getLogger(__name__)

# ---------- PDF ----------

def load_pdf_text(path: str, max_pages: Optional[int] = None) -> str:
    reader = PdfReader(path)
    pages = reader.pages[:max_pages] if max_pages else reader.pages
    texts = []
    for i, p in enumerate(pages):
        try:
            texts.append(p.extract_text() or "")
        except Exception as e:
            log.warning("Page %d could not be extracted: %s", i + 1, e)
    return "\n\n".join(texts)

# ---------- Heuristics (English-only) ----------

def quick_regex_peek_for_inclusions(text: str) -> Optional[int]:
    patterns = [
        r"included\s*\(\s*n\s*=\s*(\d+)\s*\)",  # "included (n=123)"
        r"included\s+(\d+)",                    # "included 123"
        r"n\s*=\s*(\d+)\s*included",            # "n=123 included"
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None

def quick_regex_countries(text: str) -> List[str]:
    """
    Try to capture country names near a 'Countries' header.
    Matches either:
      Countries\n<one or more lines until a blank line>
    or    Countries: A, B, C
    Returns a de-duplicated list preserving order.
    """
    results: List[str] = []

    block_pat = re.compile(r"(?im)^\s*Countries\s*\n(?P<body>(?:.+\n)+?)(?:\n\s*\n|$)")
    inline_pat = re.compile(r"(?im)^\s*Countries\s*:\s*(?P<body>.+)$")

    def split_candidates(s: str) -> List[str]:
        parts = re.split(r"[\n,;]+", s)
        out = []
        for t in parts:
            t = t.strip()
            if t:
                out.append(t)
        return out

    for m in block_pat.finditer(text):
        for c in split_candidates(m.group("body")):
            if c not in results:
                results.append(c)

    for m in inline_pat.finditer(text):
        for c in split_candidates(m.group("body")):
            if c not in results:
                results.append(c)

    return results

# ---------- NuExtract prompt builders ----------

def build_prompt_n_included(paper_text: str) -> str:
    template = '{"n_included": "integer"}'
    return "# Template:\n" + template + "\n# Context:\n" + paper_text

def build_prompt_n_included_and_countries(paper_text: str) -> str:
    template = '{"n_included": "integer", "countries": ["verbatim-string"]}'
    return "# Template:\n" + template + "\n# Context:\n" + paper_text

# ---------- LLM helpers ----------

def _json_load_stripping_fences(s: str):
    import json
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```json\n|^```\n|```$", "", s, flags=re.IGNORECASE | re.MULTILINE)
    return json.loads(s)

def _chunk_text(txt: str, max_chars: int = 40000):
    start = 0
    L = len(txt)
    while start < L:
        end = min(L, start + max_chars)
        if end < L:
            nl = txt.rfind("\n\n", start, end)
            if nl != -1 and nl > start + 2000:
                end = nl
        yield txt[start:end]
        start = end

def _call_llm(
    client: OpenAICompatibleClient,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    grammar,
) -> str:
    # Keep payload minimal for compatibility with llama.cpp builds
    return client.chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=None,
        grammar=(grammar if grammar else None),
    )

# ---------- Public API ----------

def extract_n_included_and_countries(
    client: OpenAICompatibleClient,
    paper_text: str,
    *,
    use_grammar: bool,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """
    Returns: {"n_included": int|null, "countries": [str]}
    """
    regex_n = quick_regex_peek_for_inclusions(paper_text)
    regex_c = quick_regex_countries(paper_text)

    messages = [
        {"role": "system", "content":
            "You are NuExtract: extract structured data as JSON only. "
            "Output MUST be a single JSON object, no prose, no markdown fences. "
            "If a field is not supported by the context, return null or []. "
            "Use only double quotes, valid UTF-8, and no trailing commas."
        },
        {"role": "user", "content": build_prompt_n_included_and_countries(paper_text[:200000])},
    ]

    data = None
    try:
        raw = _call_llm(
            client, messages, temperature, max_tokens,
            GRAMMAR_JSON_INT_OR_NULL if use_grammar else None
        )
        data = _json_load_stripping_fences(raw)
    except Exception as e:
        log.warning("LLM call failed on full text: %s; falling back to chunking.", e)

    if data is None or (data.get("n_included") is None and not data.get("countries")):
        vals_n: List[Optional[int]] = []
        vals_c: List[str] = []
        for part in _chunk_text(paper_text, max_chars=40000):
            try:
                raw = _call_llm(
                    client,
                    [
                        {"role": "system", "content":
                            "You are NuExtract: extract structured data as JSON only. "
                            "Output MUST be a single JSON object, no prose, no markdown fences. "
                            "If a field is not supported by the context, return null or []. "
                            "Use only double quotes, valid UTF-8, and no trailing commas."
                        },
                        {"role": "user", "content": build_prompt_n_included_and_countries(part)},
                    ],
                    temperature,
                    max_tokens,
                    GRAMMAR_JSON_INT_OR_NULL if use_grammar else None
                )
                d = _json_load_stripping_fences(raw)
                vals_n.append(d.get("n_included"))
                for c in d.get("countries") or []:
                    if isinstance(c, str) and c and c not in vals_c:
                        vals_c.append(c)
            except Exception:
                vals_n.append(None)

        non_null = [v for v in vals_n if isinstance(v, int)]
        best_n = None
        if non_null:
            from collections import Counter
            best_n = Counter(non_null).most_common(1)[0][0]
        if best_n is None:
            best_n = regex_n

        if not vals_c and regex_c:
            vals_c = regex_c

        return {"n_included": best_n, "countries": vals_c}

    # Integrate simple heuristics if model returned null/empty
    if data.get("n_included") is None and regex_n is not None:
        data["n_included"] = regex_n
    if (not data.get("countries")) and regex_c:
        data["countries"] = regex_c

    # Deduplicate countries
    if isinstance(data.get("countries"), list):
        seen: set[str] = set()
        dedup: List[str] = []
        for s in data["countries"]:
            if isinstance(s, str) and s not in seen:
                seen.add(s)
                dedup.append(s)
        data["countries"] = dedup

    return {"n_included": data.get("n_included"), "countries": data.get("countries") or []}

def extract_n_included(
    client: OpenAICompatibleClient,
    paper_text: str,
    *,
    use_grammar: bool,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """
    Backward-compatible single-field extractor.
    """
    out = extract_n_included_and_countries(
        client, paper_text,
        use_grammar=use_grammar,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return {"n_included": out.get("n_included")}
