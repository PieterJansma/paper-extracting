from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional, Dict, Any, List, Tuple, Union

from pypdf import PdfReader

from llm_client import OpenAICompatibleClient

try:
    from .llm_grammar import GRAMMAR_JSON_INT_OR_NULL
except Exception:
    GRAMMAR_JSON_INT_OR_NULL = None  # type: ignore

log = logging.getLogger(__name__)

SYSTEM_EXTRACT_MSG = (
    "You are `Qwen`. Extract structured data as JSON only. "
    "Do not output markdown fences or explanations."
)
OCR_FALLBACK_MIN_CHARS = 3000
LOW_QUALITY_ALNUM_RATIO = 0.35


# ==============================================================================
# PDF Handling
# ==============================================================================

def load_pdf_text(path: str, max_pages: Optional[int] = None) -> str:
    """
    Load text from PDF, optionally limiting pages.

    OCR fallback is attempted only when extracted text is very short
    (< OCR_FALLBACK_MIN_CHARS), because OCR is much slower.
    """
    if not path:
        return ""
    text = _load_pdf_text_pypdf(path, max_pages=max_pages)
    if not _needs_text_fallback(text):
        return text

    # First fallback: alternative pypdf extraction mode (no external tools required).
    log.warning(
        "Primary extraction looks weak (%d chars, alnum_ratio=%.2f). "
        "Trying pypdf layout fallback for %s.",
        len(text.strip()),
        _alnum_ratio(text),
        path,
    )
    layout_text = _load_pdf_text_pypdf(path, max_pages=max_pages, layout_mode=True)
    text = _pick_better_text(text, layout_text, "default", "layout")
    if not _needs_text_fallback(text):
        return text

    # OCR fallback is intentionally conservative because it is expensive
    # and depends on external binaries.
    log.warning(
        "Text still weak after layout fallback (%d chars, alnum_ratio=%.2f). "
        "Trying OCR fallback for %s.",
        len(text.strip()),
        _alnum_ratio(text),
        path,
    )
    ocr_text = _load_pdf_text_with_ocr(path, max_pages=max_pages)
    return _pick_better_text(text, ocr_text, "pre_ocr", "ocr")


def _alnum_ratio(text: str) -> float:
    t = text.strip()
    if not t:
        return 0.0
    alnum = sum(1 for ch in t if ch.isalnum())
    return alnum / len(t)


def _text_quality_score(text: str) -> float:
    t = text.strip()
    if not t:
        return 0.0
    alnum = sum(1 for ch in t if ch.isalnum())
    words = re.findall(r"[A-Za-z]{2,}", t)
    unique_words = min(len(set(w.lower() for w in words)), 2000)
    # Prefer text with real lexical content over symbol-heavy noise.
    return float(alnum) + 10.0 * float(len(words)) + 2.0 * float(unique_words)


def _needs_text_fallback(text: str) -> bool:
    t = text.strip()
    if len(t) < OCR_FALLBACK_MIN_CHARS:
        return True
    if _alnum_ratio(t) < LOW_QUALITY_ALNUM_RATIO:
        return True
    return False


def _pick_better_text(current: str, candidate: str, current_name: str, candidate_name: str) -> str:
    if not candidate or not candidate.strip():
        return current

    cur_score = _text_quality_score(current)
    cand_score = _text_quality_score(candidate)
    if cand_score > cur_score:
        log.info(
            "Text fallback improved quality (%s -> %s): chars %d -> %d, score %.1f -> %.1f.",
            current_name,
            candidate_name,
            len(current.strip()),
            len(candidate.strip()),
            cur_score,
            cand_score,
        )
        return candidate
    return current


def _load_pdf_text_pypdf(
    path: str,
    max_pages: Optional[int] = None,
    *,
    layout_mode: bool = False,
) -> str:
    try:
        reader = PdfReader(path)
    except Exception as e:
        log.error("Failed to read PDF %r: %s", path, e)
        return ""

    pages = reader.pages[:max_pages] if max_pages else reader.pages
    texts: List[str] = []
    for i, p in enumerate(pages):
        try:
            if layout_mode:
                try:
                    extracted = p.extract_text(extraction_mode="layout")
                except TypeError:
                    extracted = p.extract_text()
            else:
                extracted = p.extract_text()
            if extracted:
                texts.append(extracted)
        except Exception as e:
            log.warning("Page %d could not be extracted: %s", i + 1, e)
    return "\n\n".join(texts)


def _load_pdf_text_with_ocr(path: str, max_pages: Optional[int] = None) -> str:
    ocrmypdf_bin = shutil.which("ocrmypdf")
    if not ocrmypdf_bin:
        log.warning("OCR fallback skipped: 'ocrmypdf' not found in PATH.")
        return ""

    tmp_out = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_out_path = tmp_out.name
    tmp_out.close()

    try:
        # --skip-text avoids re-OCR on text PDFs while still handling scanned pages.
        cmd = [
            ocrmypdf_bin,
            "--skip-text",
            "--rotate-pages",
            "--deskew",
            "--quiet",
            path,
            tmp_out_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return _load_pdf_text_pypdf(tmp_out_path, max_pages=max_pages)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or "").strip()
        log.warning("OCR fallback failed for %s: %s", path, err[:500])
        return ""
    except Exception as e:
        log.warning("OCR fallback failed for %s: %s", path, e)
        return ""
    finally:
        try:
            os.remove(tmp_out_path)
        except Exception:
            pass


# ==============================================================================
# JSON parsing helpers
# ==============================================================================

def _strip_markdown_fences(s: str) -> str:
    if not s:
        return ""
    t = s.strip()

    # remove leading ```lang
    if "```" in t:
        t = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*\n", "", t)
        # remove trailing ```
        t = re.sub(r"\n\s*```\s*$", "", t)
        t = t.strip()
    return t


def _json_load_stripping_fences(s: str) -> Dict[str, Any]:
    """Parse JSON from LLM output, handling markdown code fences."""
    if not s:
        return {}
    t = _strip_markdown_fences(s)

    # Sometimes models prepend text. Try to find the first JSON object.
    # (Conservatief: alleen objecten, geen arrays.)
    if not t.startswith("{"):
        m = re.search(r"\{", t)
        if m:
            t = t[m.start():].strip()

    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass

    # Common LLM issue: trailing commas before } or ]
    t_no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", t)
    try:
        return json.loads(t_no_trailing_commas)
    except json.JSONDecodeError as e:
        salvaged = _salvage_top_level_list_output(t)
        if salvaged:
            log.warning(
                "JSON decode failed (%s), salvaged %d items from partial output.",
                e,
                sum(len(v) for v in salvaged.values() if isinstance(v, list)),
            )
            return salvaged
        # Dit is belangrijk om te zien waarom passes "leeg" lijken.
        log.warning("JSON decode failed (%s). Raw head: %r", e, t[:400])
        return {}


def _split_complete_json_objects(s: str) -> List[str]:
    """
    Extract complete top-level JSON object snippets from a string segment.
    Robust to braces in quoted strings.
    """
    objs: List[str] = []
    depth = 0
    start = -1
    in_str = False
    esc = False

    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
            continue

        if ch == "\"":
            in_str = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    objs.append(s[start:i + 1])
                    start = -1
    return objs


def _salvage_top_level_list_output(raw: str) -> Dict[str, Any]:
    """
    Try to salvage partial outputs for templates with one top-level list key.
    Example: {"collection_events":[{...},{...}, ...truncated}
    """
    if not raw:
        return {}

    candidate_keys = [
        "collection_events",
        "subpopulations",
        "datasets",
        "samplesets",
        "organisations_involved",
        "people_involved",
        "publications",
        "documentation",
    ]

    for key in candidate_keys:
        key_pat = f"\"{key}\""
        pos = raw.find(key_pat)
        if pos < 0:
            continue
        lb = raw.find("[", pos)
        if lb < 0:
            continue
        arr_segment = raw[lb + 1:]
        obj_snippets = _split_complete_json_objects(arr_segment)
        items: List[Dict[str, Any]] = []
        for snippet in obj_snippets:
            try:
                obj = json.loads(snippet)
                if isinstance(obj, dict):
                    items.append(obj)
            except Exception:
                continue
        if items:
            return {key: items}
    return {}


# ==============================================================================
# NuExtract prompt building
# ==============================================================================

def _build_nuextract_prompt(
    template_json: str,
    instructions: Optional[str],
    paper_text: str,
) -> str:
    """Construct the prompt strictly following NuExtract format."""
    template_json = _sanitize_template_json(template_json)
    instr = (instructions or "").strip()

    blocks: List[str] = []
    if template_json:
        blocks += ["# Template:", template_json]
    if instr:
        blocks += ["# Instructions:", instr]
    if paper_text:
        blocks += ["# Context:", paper_text]

    return "\n".join(blocks)


def build_context_prefix_messages(paper_text: str) -> List[Dict[str, str]]:
    """
    Prefix messages that can be reused across passes to benefit from prompt caching.
    """
    return [
        {"role": "system", "content": SYSTEM_EXTRACT_MSG},
        {"role": "user", "content": f"# Context:\n{paper_text}"},
    ]


def _split_text_chunks(
    text: str,
    chunk_size_chars: int,
    overlap_chars: int,
    max_chunks: Optional[int] = None,
) -> List[str]:
    """
    Split long context text into overlapping chunks.
    Prefers splitting near newline/space to avoid cutting sentences mid-way.
    """
    if not text:
        return []

    size = max(2000, int(chunk_size_chars))
    overlap = max(0, min(int(overlap_chars), size // 2))

    chunks: List[str] = []
    n = len(text)
    start = 0
    while start < n:
        if max_chunks is not None and max_chunks > 0 and len(chunks) >= max_chunks:
            break

        end = min(n, start + size)
        if end < n:
            lower_bound = start + int(size * 0.6)
            cut_nl = text.rfind("\n", lower_bound, end)
            cut_sp = text.rfind(" ", lower_bound, end)
            cut = max(cut_nl, cut_sp)
            if cut > start + 1000:
                end = cut

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    if not chunks:
        return [text]
    return chunks


def _has_payload_value(obj: Any) -> bool:
    if obj is None:
        return False
    if isinstance(obj, str):
        return obj.strip() != ""
    if isinstance(obj, list):
        return any(_has_payload_value(x) for x in obj)
    if isinstance(obj, dict):
        return any(_has_payload_value(v) for v in obj.values())
    return True


# ==============================================================================
# Schema completion + type normalization
# ==============================================================================

SchemaSpec = Dict[str, Tuple[str, Any]]
# where Tuple is (kind, prototype) where kind in {"scalar","list","dict"}


def _sanitize_template_json(template_json: Optional[str]) -> str:
    """Normalize hidden characters that frequently break JSON parsing in TOML multi-line strings."""
    if not template_json:
        return ""
    return (
        template_json
        .replace("\ufeff", "")   # BOM
        .replace("\u00a0", " ")  # non-breaking space
        .replace("\ufffd", " ")  # replacement char (from broken encodings)
        .strip()
    )


def _parse_template_schema(template_json: Optional[str]) -> SchemaSpec:
    """
    Parse the template JSON into a shallow schema:
    - scalar -> default None
    - list   -> default []
    - dict   -> default {}
    We only enforce top-level keys (matches your use-case).
    """
    template_json = _sanitize_template_json(template_json)
    if not template_json:
        return {}

    try:
        tpl = json.loads(template_json)
    except Exception as e:
        log.warning("Template JSON is not valid JSON (%s). No schema completion applied.", e)
        return {}

    if not isinstance(tpl, dict):
        return {}

    schema: SchemaSpec = {}
    for k, v in tpl.items():
        if isinstance(v, list):
            schema[k] = ("list", v)
        elif isinstance(v, dict):
            schema[k] = ("dict", v)
        else:
            schema[k] = ("scalar", v)
    return schema


def _apply_schema_defaults(parsed: Dict[str, Any], schema: SchemaSpec) -> Dict[str, Any]:
    """
    Ensure output always contains all top-level keys from template schema.
    Missing keys get defaults:
      - scalar -> None
      - list   -> []
      - dict   -> {}
    Also preserves any extra keys returned by the model.
    """
    if not schema:
        return parsed

    out: Dict[str, Any] = {}

    # fill expected keys
    for k, (kind, _proto) in schema.items():
        if k in parsed:
            out[k] = parsed[k]
        else:
            if kind == "list":
                out[k] = []
            elif kind == "dict":
                out[k] = {}
            else:
                out[k] = None

    # keep extras
    for k, v in parsed.items():
        if k not in out:
            out[k] = v

    return out


def _normalize_values(obj: Any) -> Any:
    """
    Light normalization:
    - "" -> None
    - "null" -> None
    - "true"/"false" -> bool
    Applies recursively for dict/list.
    """
    if isinstance(obj, dict):
        return {k: _normalize_values(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_normalize_values(v) for v in obj]

    if isinstance(obj, str):
        s = obj.strip()
        if s == "":
            return None
        if s.lower() == "null":
            return None
        if s.lower() == "true":
            return True
        if s.lower() == "false":
            return False
        return obj

    return obj


# ==============================================================================
# Merge logic
# ==============================================================================

def _merge_json_results(acc: Dict[str, Any], cur: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge JSON results.
    - Keep explicit nulls so schema stays complete across passes/papers.
    - Never overwrite an existing non-null value with null.
    - Lists are concatenated then deduped (case-insensitive for strings).
    """
    if not acc:
        return dict(cur)
    if not cur:
        return acc

    for k, v in cur.items():
        if v is None:
            # keep null only if key not present at all
            if k not in acc:
                acc[k] = None
            continue

        if k not in acc or acc[k] is None:
            acc[k] = v
            continue

        existing = acc[k]

        if isinstance(existing, list) and isinstance(v, list):
            combined = existing + v
            seen = set()
            deduped = []
            for item in combined:
                if isinstance(item, str):
                    key = item.lower().strip()
                else:
                    key = str(item)
                if key not in seen:
                    seen.add(key)
                    deduped.append(item)
            acc[k] = deduped
        else:
            # last write wins for scalars/dicts
            acc[k] = v

    return acc


# ==============================================================================
# Main Extraction Function
# ==============================================================================

def extract_fields(
    client: OpenAICompatibleClient,
    paper_text: str,
    *,
    template_json: Optional[str] = None,
    instructions: Optional[str] = None,
    use_grammar: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    prefix_messages: Optional[List[Dict[str, str]]] = None,
    cache_prompt: bool = False,
    timeout: int = 600,
    chunking_enabled: bool = True,
    long_text_threshold_chars: int = 60000,
    chunk_size_chars: int = 45000,
    chunk_overlap_chars: int = 4000,
    max_chunks: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a single extraction pass using the LLM."""
    if not paper_text:
        return {}

    schema = _parse_template_schema(template_json)

    def _run_single(
        context_text: str,
        *,
        use_prefix_messages: bool,
        apply_schema_defaults: bool,
    ) -> Tuple[Dict[str, Any], bool, bool]:
        # Returns (result, has_payload, had_llm_error)
        prompt = _build_nuextract_prompt(
            template_json or "",
            instructions,
            "" if use_prefix_messages else context_text,
        )

        log.info(
            "Prompt size: %d chars (max_tokens=%d, temp=%.2f, cache_prompt=%s, prefix=%s)",
            len(prompt),
            int(max_tokens),
            float(temperature),
            bool(cache_prompt),
            bool(use_prefix_messages),
        )

        if use_prefix_messages and prefix_messages:
            messages = list(prefix_messages) + [{"role": "user", "content": prompt}]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_EXTRACT_MSG},
                {"role": "user", "content": prompt},
            ]

        had_llm_error = False
        try:
            raw_response = client.chat(
                messages,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
                response_format={"type": "json_object"},
                grammar=(GRAMMAR_JSON_INT_OR_NULL if use_grammar else None),
                extra_body={"cache_prompt": True} if cache_prompt else None,
                timeout=int(timeout),
            )
        except Exception as e:
            log.error("LLM Call failed: %s", e)
            had_llm_error = True
            raw_response = ""

        parsed = _json_load_stripping_fences(raw_response)

        # If model returned invalid JSON, retry once with stricter JSON-object enforcement.
        if not parsed and not had_llm_error:
            log.warning("Invalid JSON output; retrying this pass once with stricter JSON mode.")
            try:
                retry_messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous answer was not valid JSON. "
                            "Return ONLY one valid JSON object that matches the template."
                        ),
                    }
                ]
                retry_raw = client.chat(
                    retry_messages,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    response_format={"type": "json_object"},
                    grammar=(GRAMMAR_JSON_INT_OR_NULL if use_grammar else None),
                    extra_body={"cache_prompt": True} if cache_prompt else None,
                    timeout=int(timeout),
                    max_retries=4,
                )
                retry_parsed = _json_load_stripping_fences(retry_raw)
                if retry_parsed:
                    parsed = retry_parsed
            except Exception as e:
                log.warning("Retry after invalid JSON failed: %s", e)

        parsed = _normalize_values(parsed)
        has_payload = _has_payload_value(parsed)
        if apply_schema_defaults:
            parsed = _apply_schema_defaults(parsed, schema)
        return parsed, has_payload, had_llm_error

    def _run_chunked() -> Dict[str, Any]:
        chunks = _split_text_chunks(
            paper_text,
            chunk_size_chars=int(chunk_size_chars),
            overlap_chars=int(chunk_overlap_chars),
            max_chunks=max_chunks,
        )
        log.warning(
            "Using chunked extraction for long paper: %d chunks (size=%d, overlap=%d).",
            len(chunks),
            int(chunk_size_chars),
            int(chunk_overlap_chars),
        )

        merged: Dict[str, Any] = {}
        payload_chunks = 0
        for idx, chunk in enumerate(chunks, start=1):
            log.info("Chunk %d/%d (%d chars)", idx, len(chunks), len(chunk))
            part, has_payload, _had_llm_error = _run_single(
                chunk,
                use_prefix_messages=False,
                apply_schema_defaults=False,
            )
            if has_payload:
                payload_chunks += 1
            merged = _merge_json_results(merged, part)

        log.info(
            "Chunked extraction complete: %d/%d chunks returned payload.",
            payload_chunks,
            len(chunks),
        )
        merged = _normalize_values(merged)
        return _apply_schema_defaults(merged, schema)

    long_text = len(paper_text) > int(long_text_threshold_chars)
    if chunking_enabled and long_text:
        return _run_chunked()

    single, has_payload, had_llm_error = _run_single(
        paper_text,
        use_prefix_messages=bool(prefix_messages),
        apply_schema_defaults=True,
    )
    if has_payload:
        return single

    # Fallback for short/medium papers when one-shot fails unexpectedly.
    if chunking_enabled and (had_llm_error or len(paper_text) > int(chunk_size_chars)):
        log.warning("One-shot extraction produced no payload; retrying with chunked mode.")
        return _run_chunked()

    return single
