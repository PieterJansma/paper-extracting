from __future__ import annotations
import logging
from typing import Optional, Dict, Any, List
from pypdf import PdfReader

from .llm_client import OpenAICompatibleClient

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

# ---------- Helpers ----------

def _json_load_stripping_fences(s: str) -> Dict[str, Any]:
    import json, re
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
    template_json = (template_json or "").strip()
    instr = (instructions or "").strip()
    blocks = ["# Template:", template_json]
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
    return client.chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=None,
        grammar=(grammar if grammar else None),
    )

def _merge_json_results(acc: Dict[str, Any], cur: Dict[str, Any]) -> Dict[str, Any]:
    from collections import Counter
    if not acc:
        return dict(cur)
    for k, v in cur.items():
        if k not in acc:
            acc[k] = v
            continue
        a = acc[k]
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
        if isinstance(a, (int, float)) or isinstance(v, (int, float)):
            if a is None:
                acc[k] = v
            elif v is None:
                pass
            else:
                acc[k] = Counter([a, v]).most_common(1)[0][0]
            continue
        if isinstance(a, str) or isinstance(v, str):
            acc[k] = a if (isinstance(a, str) and a) else v
            continue
        if a is None and v is not None:
            acc[k] = v
    return acc

# ---------- Defaults (edit these as you add fields) ----------

DEFAULT_TEMPLATE = '{"n_included": "integer", "countries": ["verbatim-string"]}'

DEFAULT_INSTRUCTIONS = (
    "Extract the number of INCLUDED participants and the list of countries where the INCLUDED participants came from.\n"
    "- Include ONLY countries for the final included cohort.\n"
    "- EXCLUDE countries for screened/excluded/eligible-but-not-included participants, non-contributing sites, and author affiliations.\n"
    '- Return each country name verbatim (e.g., "United Kingdom of Great Britain and Northern Ireland (the)").\n'
    "- Deduplicate; order does not matter."
)

RETRY_INSTRUCTIONS_SUFFIX = (
    "If the context contains country names for the included cohort, you MUST return them in `countries`.\n"
    "Do not return an empty list when such country names are present.\n"
    "If none are present for the included cohort, return an empty list.\n"
    "Example:\n"
    "# Context:\n"
    "Countries\n"
    "United Kingdom of Great Britain and Northern Ireland (the)\n"
    "# Expected JSON:\n"
    '{"n_included": null, "countries": ["United Kingdom of Great Britain and Northern Ireland (the)"]}'
)

def _windows_by_keywords(txt: str, keywords: List[str], half_window: int = 1400, max_hits: int = 10) -> List[str]:
    t_low = txt.lower()
    hits = []
    for kw in keywords:
        start = 0
        kw_low = kw.lower()
        while True:
            idx = t_low.find(kw_low, start)
            if idx == -1:
                break
            left = max(0, idx - half_window)
            right = min(len(txt), idx + len(kw) + half_window)
            hits.append((left, right))
            start = idx + len(kw_low)
            if len(hits) >= max_hits:
                break
        if len(hits) >= max_hits:
            break
    if not hits:
        return []
    hits.sort()
    merged = []
    cur_l, cur_r = hits[0]
    for l, r in hits[1:]:
        if l <= cur_r + 50:
            cur_r = max(cur_r, r)
        else:
            merged.append((cur_l, cur_r))
            cur_l, cur_r = l, r
    merged.append((cur_l, cur_r))
    return [txt[l:r] for (l, r) in merged]

COUNTRY_KEYWORDS = [
    "countries",
    "country",
    "participating countries",
    "participants were from",
    "recruited from",
    "enrolled from",
    "came from",
]

# ---------- Single public function ----------

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
    Returns the JSON object produced by NuExtract, merged across chunks/windows.
    """
    tpl = (template_json or DEFAULT_TEMPLATE)
    instr = (instructions or DEFAULT_INSTRUCTIONS)

    system_msg = (
        "You are NuExtract: extract structured data as JSON only. "
        "Output MUST be a single JSON object, no prose, no markdown fences. "
        "If a field is not supported by the context, return null or []. "
        "Use only double quotes, valid UTF-8, and no trailing commas."
    )

    # Pass 1: full text (capped)
    full_prompt = _build_nuextract_prompt(tpl, instr, paper_text[:200000])
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": full_prompt},
    ]

    data: Dict[str, Any] = {}
    try:
        raw = _call_llm_minimal(
            client, messages, temperature, max_tokens,
            GRAMMAR_JSON_INT_OR_NULL if use_grammar else None
        )
        data = _json_load_stripping_fences(raw)
    except Exception as e:
        log.warning("Full-text LLM call failed (%s).", e)
        data = {}

    # If countries missing/empty, try small windows around likely phrases
    if isinstance(data, dict) and not data.get("countries"):
        windows = _windows_by_keywords(paper_text, COUNTRY_KEYWORDS, half_window=1400, max_hits=10)
        for w in windows:
            retry_prompt = _build_nuextract_prompt(
                tpl,
                (instr + "\n" + RETRY_INSTRUCTIONS_SUFFIX) if instr else RETRY_INSTRUCTIONS_SUFFIX,
                w
            )
            retry_messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": retry_prompt},
            ]
            try:
                raw2 = _call_llm_minimal(
                    client, retry_messages, temperature, max_tokens,
                    GRAMMAR_JSON_INT_OR_NULL if use_grammar else None
                )
                d2 = _json_load_stripping_fences(raw2)
                if isinstance(d2, dict):
                    data = _merge_json_results(data, d2)
            except Exception as ex:
                log.debug("Window retry failed: %s", ex)

    if isinstance(data, dict) and data:
        return data

    # Pass 2: chunk fallback (merge results)
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
            if isinstance(cur, dict) and not cur.get("countries"):
                wins = _windows_by_keywords(part, COUNTRY_KEYWORDS, half_window=1400, max_hits=6)
                for w in wins:
                    rprompt = _build_nuextract_prompt(
                        tpl, (instr + "\n" + RETRY_INSTRUCTIONS_SUFFIX), w
                    )
                    rmsgs = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": rprompt},
                    ]
                    try:
                        raw_r = _call_llm_minimal(
                            client, rmsgs, temperature, max_tokens,
                            GRAMMAR_JSON_INT_OR_NULL if use_grammar else None
                        )
                        cur_r = _json_load_stripping_fences(raw_r)
                        if isinstance(cur_r, dict):
                            cur = _merge_json_results(cur if isinstance(cur, dict) else {}, cur_r)
                    except Exception as rex:
                        log.debug("Chunk window retry failed: %s", rex)

            merged = _merge_json_results(merged, cur if isinstance(cur, dict) else {})
        except Exception as ex:
            log.debug("Chunk failed: %s", ex)

    return merged
