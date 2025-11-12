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

def _build_nuextract_prompt(template_json: str, instructions: Optional[str], paper_text: str) -> str:
    # NuExtract expects: # Template:\n{json}\n# Context:\n{text}
    # We optionally add a minimal # Instructions: block.
    template_json = (template_json or "").strip()
    instr = (instructions or "").strip()
    blocks = [
        "# Template:",
        template_json,
    ]
    if instr:
        blocks += ["# Instructions:", instr]
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
    - numbers/strings: prefer majority/first non-null
    - null handling: keep non-null if any seen
    """
    from collections import Counter

    if not acc:
        return dict(cur)

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
                key = ("s", item) if isinstance(item, str) else ("n", item) if isinstance(item, (int, float)) else ("o", repr(item))
                if key not in seen:
                    seen.add(key)
                    merged.append(item)
            acc[k] = merged
            continue

        # numeric: pick majority among [a, v]; tie -> keep a
        if isinstance(a, (int, float)) or isinstance(v, (int, float)):
            if a is None:
                acc[k] = v
            elif v is None:
                pass
            else:
                pick = Counter([a, v]).most_common(1)[0][0]
                acc[k] = pick
            continue

        # strings: prefer non-empty
        if isinstance(a, str) or isinstance(v, str):
            acc[k] = a if (isinstance(a, str) and a) else v
            continue

        # fallback: prefer non-null
        if a is None and v is not None:
            acc[k] = v

    return acc

# ---------- Light heuristics specifically for countries ----------

_COUNTRIES_BLOCK = re.compile(r"(?im)^\s*Countries\s*\n(?P<body>(?:.+\n)+?)(?:\n\s*\n|$)")
_COUNTRIES_INLINE = re.compile(r"(?im)^\s*Countries\s*:\s*(?P<body>.+)$")

def _regex_countries(text: str) -> List[str]:
    def split_candidates(s: str) -> List[str]:
        parts = re.split(r"[\n,;]+", s)
        out = []
        for t in parts:
            t = t.strip()
            if t:
                out.append(t)
        return out

    results: List[str] = []
    for m in _COUNTRIES_BLOCK.finditer(text):
        for c in split_candidates(m.group("body")):
            if c not in results:
                results.append(c)
    for m in _COUNTRIES_INLINE.finditer(text):
        for c in split_candidates(m.group("body")):
            if c not in results:
                results.append(c)
    return results

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
    template_json: Optional[str] = None,
    instructions: Optional[str] = None,
    use_grammar: bool,
    temperature: float,
    max_tokens: int,
    chunk_chars: int = 40000,
) -> Dict[str, Any]:
    """
    Generic extractor. Edit `template_json` (add fields one by one) and/or `instructions`.
    Returns the JSON object produced by NuExtract, optionally merged across chunks.
    """
    tpl = (template_json or DEFAULT_TEMPLATE)
    instr = (instructions or DEFAULT_INSTRUCTIONS)

    system_msg = (
        "You are NuExtract: extract structured data as JSON only. "
        "Output MUST be a single JSON object, no prose, no markdown fences. "
        "If a field is not supported by the context, return null or []. "
        "Use only double quotes, valid UTF-8, and no trailing commas."
    )

    # Try full text once (cap to avoid absurdly long prompts).
    full_prompt = _build_nuextract_prompt(tpl, instr, paper_text[:200000])
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
    except Exception as e:
        log.warning("Full-text LLM call failed (%s). Falling back to chunking.", e)
        data = {}

    # If countries empty, try regex to recover obvious "Countries" lists.
    if isinstance(data, dict):
        if ("countries" not in data) or (not data.get("countries")):
            rx = _regex_countries(paper_text)
            if rx:
                data["countries"] = rx

    if data:
        return data

    # Chunk fallback: run over chunks and merge results generically.
    merged: Dict[str, Any] = {}
    for part in _chunk_text(paper_text, max_chars=chunk_chars):
        part_prompt = _build_nuextract_prompt(tpl, instr, part)
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
            # if this chunk missed countries, try regex on the chunk too
            if isinstance(cur, dict) and (not cur.get("countries")):
                rx = _regex_countries(part)
                if rx:
                    cur["countries"] = rx
            merged = _merge_json_results(merged, cur if isinstance(cur, dict) else {})
        except Exception as ex:
            log.debug("Chunk failed: %s", ex)

    return merged
