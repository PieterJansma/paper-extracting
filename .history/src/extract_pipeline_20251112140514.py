from __future__ import annotations
import logging
import re
from typing import Optional, Dict, Any, List
from pypdf import PdfReader

from .llm_client import OpenAICompatibleClient

# Optional grammar; safe to ignore if your server rejects it.
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

# ---------- Generic helpers ----------

def _json_load_stripping_fences(s: str) -> Dict[str, Any]:
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

def _build_nuextract_prompt(template_json: str, instructions: str, paper_text: str) -> str:
    # NuExtract expects: # Template:\n{json}\n# Context:\n{text}
    # We add a minimal # Instructions: block to disambiguate when needed.
    blocks = [
        "# Template:",
        template_json,
    ]
    if instructions.strip():
        blocks += ["# Instructions:", instructions.strip()]
    blocks += ["# Context:", paper_text]
    return "\n".join(blocks)

def _call_llm_minimal(
    client: OpenAICompatibleClient,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    grammar,
) -> str:
    # Keep payload minimal for llama.cpp compatibility (no response_format).
    return client.chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=None,
        grammar=(grammar if grammar else None),
    )

def _merge_json_results(acc: Dict[str, Any], cur: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic merger:
    - lists: union (order preserved by first appearance)
    - numbers/strings: prefer majority non-null if we see the same type multiple times; else first non-null
    - null handling: keep non-null if any seen
    """
    from collections import Counter

    if not acc:
        return cur

    for k, v in cur.items():
        if k not in acc:
            acc[k] = v
            continue

        a = acc[k]

        # list → union
        if isinstance(a, list) or isinstance(v, list):
            left = a if isinstance(a, list) else ([] if a is None else [a])
            right = v if isinstance(v, list) else ([] if v is None else [v])
            seen = set()
            merged = []
            for item in left + right:
                if item is None:
                    continue
                if isinstance(item, str):
                    key = ("s", item)
                elif isinstance(item, (int, float)):
                    key = ("n", item)
                else:
                    key = ("o", repr(item))
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
            acc[k] = merged
            continue

        # numeric: pick majority non-null if possible
        if isinstance(a, (int, float)) or isinstance(v, (int, float)):
            # collapse to majority among previous + current if we store a small history
            # Here: simple rule — if acc is null use v; else keep acc unless v equals a
            if a is None:
                acc[k] = v
            elif v is None:
                pass
            else:
                # if they differ, pick the one that appears more often in the pair list (tie → keep existing)
                c = Counter([a, v])
                pick, _ = c.most_common(1)[0]
                acc[k] = pick
            continue

        # strings: prefer non-empty
        if isinstance(a, str) or isinstance(v, str):
            acc[k] = a if (a and a != "null") else v
            continue

        # fallback: prefer non-null
        if a is None and v is not None:
            acc[k] = v

    return acc

# ---------- Defaults you can edit as you add variables ----------

DEFAULT_TEMPLATE = '{"n_included": "integer", "countries": ["verbatim-string"]}'

DEFAULT_INSTRUCTIONS = (
    "Extract the number of INCLUDED participants and the list of countries where the INCLUDED participants came from.\n"
    "- Include ONLY countries for the final included cohort.\n"
    "- EXCLUDE countries for screened/excluded/eligible-but-not-included participants, non-contributing sites, and author affiliations.\n"
    '- Return each country name verbatim (e.g., "United Kingdom of Great Britain and Northern Ireland (the)").\n'
    "- Deduplicate; order does not matter."
)

# ---------- Single public function you reuse by editing the template/instructions ----------

def extract_fields(
    client: OpenAICompatibleClient,
    paper_text: str,
    *,
    template_json: str = DEFAULT_TEMPLATE,
    instructions: str = DEFAULT_INSTRUCTIONS,
    use_grammar: bool,
    temperature: float,
    max_tokens: int,
    chunk_chars: int = 40000,
) -> Dict[str, Any]:
    """
    Generic extractor. Edit `template_json` (add fields one by one) and/or `instructions`.
    Returns the JSON object produced by NuExtract, optionally merged across chunks.
    """
    system_msg = (
        "You are NuExtract: extract structured data as JSON only. "
        "Output MUST be a single JSON object, no prose, no markdown fences. "
        "If a field is not supported by the context, return null or []. "
        "Use only double quotes, valid UTF-8, and no trailing commas."
    )

    # Try full text once (cap to avoid absurdly long prompts).
    full_prompt = _build_nuextract_prompt(template_json, instructions, paper_text[:200000])
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": full_prompt},
    ]

    try:
        raw = _call_llm_minimal(
            client, messages, temperature, max_tokens,
            GRAMMAR_JSON_INT_OR_NULL if use_grammar else None
        )
        data = _json_load_stripping_fences(raw)
        return data
    except Exception as e:
        msg = str(e)
        log.warning("Full-text LLM call failed (%s). Falling back to chunking.", msg)

    # Chunk fallback: run over chunks and merge results generically.
    merged: Dict[str, Any] = {}
    for part in _chunk_text(paper_text, max_chars=chunk_chars):
        part_prompt = _build_nuextract_prompt(template_json, instructions, part)
        part_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": part_prompt},
        ]
        try:
            raw = _call_llm_minimal(
                client, part_messages, temperature, max_tokens,
                GRAMMAR_JSON_INT_OR_NULL if use_grammar else None
            )
            cur = _json_load_stripping_fences(raw)
            merged = _merge_json_results(merged, cur)
        except Exception as ex:
            log.debug("Chunk failed: %s", ex)

    return merged
