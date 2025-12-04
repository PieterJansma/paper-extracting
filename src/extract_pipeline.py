from __future__ import annotations
import logging
import re
import json
from typing import Optional, Dict, Any, List
from pypdf import PdfReader

# GOED (zonder punt)
from llm_client import OpenAICompatibleClient
# Probeer grammar te laden, maar faal niet als het mist
try:
    from .llm_grammar import GRAMMAR_JSON_INT_OR_NULL
except Exception:
    GRAMMAR_JSON_INT_OR_NULL = None  # type: ignore

log = logging.getLogger(__name__)

# ---------- PDF Handling ----------

def load_pdf_text(path: str, max_pages: Optional[int] = None) -> str:
    """Load text from PDF, optionally limiting pages."""
    if not path:
        return ""
    try:
        reader = PdfReader(path)
    except Exception as e:
        log.error(f"Failed to read PDF: {e}")
        return ""

    pages = reader.pages[:max_pages] if max_pages else reader.pages
    texts: List[str] = []
    
    for i, p in enumerate(pages):
        try:
            extracted = p.extract_text()
            if extracted:
                texts.append(extracted)
        except Exception as e:
            log.warning("Page %d could not be extracted: %s", i + 1, e)
            
    full_text = "\n\n".join(texts)
    return full_text

# ---------- Helpers ----------

def _json_load_stripping_fences(s: str) -> Dict[str, Any]:
    """Parse JSON from LLM output, handling markdown code fences."""
    if not s:
        return {}
    s = s.strip()
    # Remove ```json ... ``` fences
    if "```" in s:
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
    
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        log.debug(f"JSON Decode Error: {e}. Raw output: {s[:100]}...")
        return {}

def _build_nuextract_prompt(
    template_json: str,
    instructions: Optional[str],
    paper_text: str,
) -> str:
    """Construct the prompt strictly following NuExtract format."""
    template_json = (template_json or "").strip()
    instr = (instructions or "").strip()
    
    blocks = []
    # NuExtract expects the template first
    if template_json:
        blocks += ["# Template:", template_json]
    # Then instructions
    if instr:
        blocks += ["# Instructions:", instr]
    # Then the context (paper text)
    if paper_text:
        blocks += ["# Context:", paper_text]
        
    return "\n".join(blocks)

def _merge_json_results(acc: Dict[str, Any], cur: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge JSON results. 
    - Lists are concatenated and deduplicated.
    - Strings/Numbers are overwritten by the new value if not None.
    """
    if not acc:
        return dict(cur)
    if not cur:
        return acc

    for k, v in cur.items():
        if v is None:
            continue # Don't overwrite existing data with null
            
        if k not in acc:
            acc[k] = v
            continue
            
        existing = acc[k]

        # Merge Lists
        if isinstance(existing, list) and isinstance(v, list):
            # Create a combined list, deduplicating by string representation
            combined = existing + v
            seen = set()
            deduped = []
            for item in combined:
                # Make simple types hashable for deduplication
                key = str(item).lower().strip() if isinstance(item, str) else str(item)
                if key not in seen:
                    seen.add(key)
                    deduped.append(item)
            acc[k] = deduped
            
        # Overwrite scalars (Pass B/C takes precedence over Pass A if they overlap)
        else:
            acc[k] = v
            
    return acc

# ---------- Main Extraction Function ----------

def extract_fields(
    client: OpenAICompatibleClient,
    paper_text: str,
    *,
    template_json: Optional[str] = None,
    instructions: Optional[str] = None,
    use_grammar: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """
    Execute a single extraction pass using the LLM.
    """
    if not paper_text:
        return {}

    # Build the strict prompt
    prompt = _build_nuextract_prompt(template_json, instructions, paper_text)
    
    # System message for stability
    system_msg = (
        "You are NuExtract. Extract structured data as JSON only. "
        "Do not output markdown fences or explanations."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    # Call LLM
    try:
        raw_response = client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            grammar=(GRAMMAR_JSON_INT_OR_NULL if use_grammar else None),
        )
    except Exception as e:
        log.error(f"LLM Call failed: {e}")
        return {}

    # Parse and return
    return _json_load_stripping_fences(raw_response)