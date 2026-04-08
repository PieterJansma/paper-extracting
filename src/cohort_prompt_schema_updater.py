from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml

from cohort_dynamic_prompts import build_dynamic_task_sections
from emx2_dynamic_runtime import build_runtime_registry, write_json, write_task_prompts_toml
from llm_client import OpenAICompatibleClient


TASK_PREFIX = "task_"
FIELD_LINE_RE = re.compile(r"^\s*-\s+`([^`]+)`:\s*(.*)$")


def _load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        data = toml.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid TOML root in {path}")
    return data


def _task_sections(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in cfg.items():
        if str(key).startswith(TASK_PREFIX) and isinstance(value, dict):
            out[str(key)] = dict(value)
    return out


def _parse_template_json(template_json: str | None) -> Any:
    if not template_json:
        return {}
    try:
        return json.loads(template_json)
    except Exception:
        return {}


def _flatten_template(node: Any, prefix: str = "") -> Dict[str, str]:
    out: Dict[str, str] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            if isinstance(value, list) and value and isinstance(value[0], dict):
                out.update(_flatten_template(value[0], f"{new_prefix}[]"))
            elif isinstance(value, dict):
                out.update(_flatten_template(value, new_prefix))
            else:
                out[new_prefix] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return out


def _template_diff(old_template_json: str, new_template_json: str) -> Dict[str, List[str]]:
    old_map = _flatten_template(_parse_template_json(old_template_json))
    new_map = _flatten_template(_parse_template_json(new_template_json))
    old_keys = set(old_map)
    new_keys = set(new_map)
    added = sorted(new_keys - old_keys)
    removed = sorted(old_keys - new_keys)
    changed = sorted(k for k in (old_keys & new_keys) if old_map[k] != new_map[k])
    return {
        "added_fields": added,
        "removed_fields": removed,
        "changed_fields": changed,
    }


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _field_lines(instructions: str | None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw_line in str(instructions or "").splitlines():
        match = FIELD_LINE_RE.match(raw_line)
        if not match:
            continue
        out[match.group(1)] = raw_line.strip()
    return out


def _extract_original_field_blocks(instructions: str | None) -> Dict[str, str]:
    lines = str(instructions or "").splitlines()
    out: Dict[str, str] = {}

    def is_divider(idx: int) -> bool:
        if idx < 0 or idx >= len(lines):
            return False
        stripped = lines[idx].strip()
        return len(stripped) >= 10 and set(stripped) == {"-"}

    i = 0
    while i + 2 < len(lines):
        if is_divider(i) and is_divider(i + 2):
            heading = lines[i + 1].strip()
            if heading:
                j = i + 3
                while j + 2 < len(lines) and not (is_divider(j) and is_divider(j + 2)):
                    j += 1
                if j + 2 >= len(lines):
                    j = len(lines)
                block = "\n".join(lines[i + 1 : j]).strip()
                heading_key = re.sub(r"\s*\(.*?\)\s*$", "", heading).strip()
                out[_normalize(heading_key)] = block
                for part in heading_key.split("/"):
                    part_key = _normalize(part)
                    if part_key:
                        out.setdefault(part_key, block)
                i = j
                continue
        i += 1
    return out


def _heading_aliases(heading: str) -> set[str]:
    heading_key = re.sub(r"\s*\(.*?\)\s*$", "", str(heading or "")).strip()
    aliases = {_normalize(heading_key)}
    for part in re.split(r"\s*/\s*", heading_key):
        part_key = _normalize(part)
        if part_key:
            aliases.add(part_key)
    return {alias for alias in aliases if alias}


def _parse_instruction_segments(instructions: str | None) -> List[Dict[str, Any]]:
    lines = str(instructions or "").splitlines()
    segments: List[Dict[str, Any]] = []

    def is_divider(idx: int) -> bool:
        if idx < 0 or idx >= len(lines):
            return False
        stripped = lines[idx].strip()
        return len(stripped) >= 10 and set(stripped) == {"-"}

    def is_top_level_bullet(idx: int) -> bool:
        if idx < 0 or idx >= len(lines):
            return False
        line = lines[idx]
        return line.startswith("- ") and not line.startswith("  ")

    cursor = 0
    i = 0
    while i < len(lines):
        if i + 2 < len(lines) and is_divider(i) and is_divider(i + 2):
            if cursor < i:
                segments.append({
                    "kind": "text",
                    "text": "\n".join(lines[cursor:i]).rstrip(),
                })
            heading = lines[i + 1].strip()
            heading_key = re.sub(r"\s*\(.*?\)\s*$", "", heading).strip()
            j = i + 3
            while j < len(lines):
                if j + 2 < len(lines) and is_divider(j) and is_divider(j + 2):
                    break
                j += 1
            block = "\n".join(lines[i + 1:j]).strip()
            segments.append({
                "kind": "field_block",
                "style": "divider",
                "heading": heading,
                "heading_key": heading_key,
                "aliases": _heading_aliases(heading_key),
                "block": block,
            })
            cursor = j
            i = j
            continue

        if is_top_level_bullet(i):
            if cursor < i:
                segments.append({
                    "kind": "text",
                    "text": "\n".join(lines[cursor:i]).rstrip(),
                })
            j = i + 1
            while j < len(lines):
                if is_top_level_bullet(j):
                    break
                if j + 2 < len(lines) and is_divider(j) and is_divider(j + 2):
                    break
                j += 1
            first_line = lines[i][2:]
            heading = first_line.split(":", 1)[0].strip()
            block = "\n".join(lines[i:j]).rstrip()
            segments.append({
                "kind": "field_block",
                "style": "bullet",
                "heading": heading,
                "heading_key": heading,
                "aliases": _heading_aliases(heading),
                "block": block,
            })
            cursor = j
            i = j
            continue

        i += 1
    if cursor < len(lines):
        segments.append({
            "kind": "text",
            "text": "\n".join(lines[cursor:]).rstrip(),
        })
    return segments


def _render_instruction_segments(segments: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for seg in segments:
        if seg.get("kind") == "text":
            text = str(seg.get("text") or "").rstrip()
            if text:
                parts.append(text)
            continue
        if seg.get("kind") == "field_block":
            block = str(seg.get("block") or "").strip()
            if block:
                if seg.get("style") == "divider":
                    parts.append("--------------------------------------------------\n" + block)
                else:
                    parts.append(block)
    return "\n\n".join(part for part in parts if part).rstrip()


def _field_path_leaf(field_path: str) -> str:
    return str(field_path).split(".")[-1].replace("[]", "")


def _find_segment_index_for_field(segments: List[Dict[str, Any]], field_path: str) -> int:
    leaf = _normalize(_field_path_leaf(field_path))
    for idx, seg in enumerate(segments):
        if seg.get("kind") != "field_block":
            continue
        aliases = set(seg.get("aliases") or set())
        if leaf in aliases:
            return idx
    return -1


def _new_field_order(instructions: str | None) -> List[str]:
    return list(_field_lines(instructions).keys())


def _insert_segment_for_field(
    segments: List[Dict[str, Any]],
    *,
    field_path: str,
    new_block: str,
    new_generated_section: Dict[str, Any],
) -> None:
    order = _new_field_order(new_generated_section.get("instructions"))
    try:
        target_idx = order.index(field_path)
    except ValueError:
        target_idx = len(order)

    insert_at = len(segments)
    for prev_field in reversed(order[:target_idx]):
        prev_idx = _find_segment_index_for_field(segments, prev_field)
        if prev_idx >= 0:
            insert_at = prev_idx + 1
            break
    else:
        for idx, seg in enumerate(segments):
            if seg.get("kind") == "field_block":
                insert_at = idx
                break

    heading = _field_path_leaf(field_path)
    heading = heading.replace("_", " ")
    style = "divider"
    neighbor_indices = []
    if insert_at - 1 >= 0:
        neighbor_indices.append(insert_at - 1)
    if insert_at < len(segments):
        neighbor_indices.append(insert_at)
    for neighbor_idx in neighbor_indices:
        neighbor = segments[neighbor_idx]
        if neighbor.get("kind") == "field_block":
            style = str(neighbor.get("style") or style)
            break
    normalized_block = _coerce_block_style(new_block, style, field_path)
    segments.insert(insert_at, {
        "kind": "field_block",
        "style": style,
        "heading": heading,
        "heading_key": heading,
        "aliases": _heading_aliases(heading),
        "block": normalized_block,
    })


def _coerce_block_style(block: str | None, style: str, field_path: str) -> str:
    text = str(block or "").strip()
    if not text:
        return ""
    if style != "divider":
        return text

    lines = text.splitlines()
    first = lines[0] if lines else ""
    match = FIELD_LINE_RE.match(first)
    if not match:
        return text

    heading = _field_path_leaf(field_path).replace("_", " ")
    rendered: List[str] = [heading]
    body = match.group(2).strip()
    if body:
        rendered.append(body)
    for raw in lines[1:]:
        rendered.append(raw[2:] if raw.startswith("  ") else raw)
    return "\n".join(part for part in rendered if part).strip()


def _changed_field_paths(
    old_section: Dict[str, Any],
    new_section: Dict[str, Any],
    diff: Dict[str, List[str]],
) -> List[str]:
    old_lines = _field_lines(old_section.get("instructions"))
    new_lines = _field_lines(new_section.get("instructions"))
    changed: set[str] = set()

    for key in set(old_lines) | set(new_lines):
        if old_lines.get(key) != new_lines.get(key):
            changed.add(key)

    line_keys = set(old_lines) | set(new_lines)

    def resolve_leaf(leaf: str) -> str:
        if leaf in line_keys:
            return leaf
        for key in sorted(line_keys):
            if key.endswith(f".{leaf}") or key.endswith(f"[]{'.' + leaf}"):
                return key
        return leaf

    for leaf in list(diff.get("added_fields") or []):
        changed.add(resolve_leaf(str(leaf)))
    for leaf in list(diff.get("removed_fields") or []):
        changed.add(resolve_leaf(str(leaf)))
    for leaf in list(diff.get("changed_fields") or []):
        changed.add(resolve_leaf(str(leaf)))

    return sorted(changed)


def _prepend_schema_change_notice(task_name: str, section: Dict[str, Any], diff: Dict[str, List[str]]) -> Dict[str, Any]:
    changed = []
    if diff["added_fields"]:
        changed.append(f"- Added fields: {', '.join(diff['added_fields'])}")
    if diff["removed_fields"]:
        changed.append(f"- Removed fields: {', '.join(diff['removed_fields'])}")
    if diff["changed_fields"]:
        changed.append(f"- Changed fields: {', '.join(diff['changed_fields'])}")

    header = [
        "SCHEMA-DRIVEN UPDATE",
        f"- This task was regenerated because the EMX2 schema changed for `{task_name}`.",
    ]
    header.extend(changed)
    header.append("")

    updated = dict(section)
    updated["instructions"] = "\n".join(header) + str(section.get("instructions") or "")
    return updated


def _merge_changed_task_into_base(
    *,
    task_name: str,
    base_section: Dict[str, Any],
    old_section: Dict[str, Any],
    new_section: Dict[str, Any],
    diff: Dict[str, List[str]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    updated = dict(base_section)
    if new_section.get("template_json"):
        updated["template_json"] = new_section.get("template_json")

    base_instructions = str(base_section.get("instructions") or "").strip()
    if not base_instructions:
        updated["instructions"] = str(new_section.get("instructions") or "").strip()
        return updated, {
            "mode": "replace_no_base_instructions",
            "fields": [],
        }

    segments = _parse_instruction_segments(base_section.get("instructions"))
    if not any(seg.get("kind") == "field_block" for seg in segments):
        updated["instructions"] = base_instructions
        return updated, {
            "mode": "preserve_unstructured_base",
            "fields": [],
        }

    old_lines = _field_lines(old_section.get("instructions"))
    new_lines = _field_lines(new_section.get("instructions"))
    changed_fields = _changed_field_paths(old_section, new_section, diff)
    field_actions: List[Dict[str, str]] = []

    for field_path in changed_fields:
        idx = _find_segment_index_for_field(segments, field_path)
        new_exists = field_path in new_lines
        old_exists = field_path in old_lines

        if old_exists and not new_exists:
            if idx >= 0:
                segments.pop(idx)
                field_actions.append({"field_path": field_path, "action": "removed"})
            continue

        if new_exists and not old_exists:
            new_block = _extract_field_block(new_section.get("instructions"), field_path)
            if new_block:
                _insert_segment_for_field(
                    segments,
                    field_path=field_path,
                    new_block=new_block,
                    new_generated_section=new_section,
                )
                field_actions.append({"field_path": field_path, "action": "added"})
            continue

        if idx >= 0:
            field_actions.append({"field_path": field_path, "action": "preserved_existing_block"})
        else:
            new_block = _extract_field_block(new_section.get("instructions"), field_path)
            if new_block:
                _insert_segment_for_field(
                    segments,
                    field_path=field_path,
                    new_block=new_block,
                    new_generated_section=new_section,
                )
                field_actions.append({"field_path": field_path, "action": "added_missing_block"})

    merged_instructions = _render_instruction_segments(segments).strip()
    updated["instructions"] = merged_instructions or base_instructions
    return updated, {
        "mode": "field_merge",
        "fields": field_actions,
        "task_name": task_name,
    }


def build_updated_prompt_cfg(
    *,
    base_prompts: Dict[str, Dict[str, Any]],
    old_generated: Dict[str, Dict[str, Any]],
    new_generated: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    report: Dict[str, Any] = {"tasks": {}}

    task_names = sorted(set(base_prompts) | set(old_generated) | set(new_generated))
    for task_name in task_names:
        base_section = base_prompts.get(task_name)
        old_section = old_generated.get(task_name, {})
        new_section = new_generated.get(task_name, {})
        diff = _template_diff(
            str(old_section.get("template_json") or ""),
            str(new_section.get("template_json") or ""),
        )
        instruction_changed = (
            str(old_section.get("instructions") or "").strip()
            != str(new_section.get("instructions") or "").strip()
        )
        changed = any(diff.values()) or instruction_changed
        report["tasks"][task_name] = {
            "changed": changed,
            "instruction_changed": instruction_changed,
            **diff,
        }

        if not changed and base_section is not None:
            out[task_name] = dict(base_section)
            continue
        if base_section is not None:
            merged_section, merge_report = _merge_changed_task_into_base(
                task_name=task_name,
                base_section=base_section,
                old_section=old_section,
                new_section=new_section,
                diff=diff,
            )
            out[task_name] = merged_section
            report["tasks"][task_name]["merge_mode"] = merge_report.get("mode")
            report["tasks"][task_name]["field_actions"] = list(merge_report.get("fields") or [])
            continue
        out[task_name] = dict(new_section)
        report["tasks"][task_name]["merge_mode"] = "new_task_generated"

    return out, report


def _load_llm_cfg(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    cfg = _load_toml(Path(path).expanduser().resolve())
    llm_cfg = cfg.get("llm")
    return dict(llm_cfg) if isinstance(llm_cfg, dict) else {}


def _build_rewrite_messages(
    *,
    task_name: str,
    base_section: Dict[str, Any] | None,
    old_section: Dict[str, Any],
    new_section: Dict[str, Any],
    diff: Dict[str, List[str]],
) -> List[Dict[str, str]]:
    changed_field_paths = _changed_field_paths(old_section, new_section, diff)
    old_prompt_blocks = _extract_original_field_blocks((base_section or {}).get("instructions", ""))
    old_generated_lines = _field_lines(old_section.get("instructions"))
    new_generated_lines = _field_lines(new_section.get("instructions"))

    changed_field_details = []
    for field_path in changed_field_paths:
        leaf = str(field_path).split(".")[-1].replace("[]", "")
        changed_field_details.append(
            {
                "field_path": field_path,
                "old_prompt_block": old_prompt_blocks.get(_normalize(leaf), ""),
                "old_generated_rule": old_generated_lines.get(field_path, ""),
                "new_generated_rule": new_generated_lines.get(field_path, ""),
            }
        )

    payload = {
        "task_name": task_name,
        "schema_diff": diff,
        "changed_field_details": changed_field_details,
        "old_template_json": old_section.get("template_json", ""),
        "new_template_json": new_section.get("template_json", ""),
        "old_instructions": (base_section or {}).get("instructions", ""),
        "deterministic_new_instructions": new_section.get("instructions", ""),
    }
    prompt = (
        "Rewrite the prompt instructions for one changed extraction task.\n"
        "Goals:\n"
        "- Keep the strong style and task-specific nuance from the old instructions when still valid.\n"
        "- Adapt the instructions to the new schema exactly.\n"
        "- Prefer minimally editing the old task instead of rewriting it from scratch.\n"
        "- For changed fields with an old prompt block, preserve that block's concrete rules, examples and edge-case wording whenever still valid.\n"
        "- For unchanged fields, keep the original detailed blocks and overall task framing.\n"
        "- Do not mention removed fields.\n"
        "- Do mention newly added fields when relevant.\n"
        "- For a newly added field that has no old block, write a new field block in the same style and level of detail as neighboring blocks in the old instructions.\n"
        "- If the old task uses divider-style field sections, keep that formatting.\n"
        "- Do not collapse a detailed original prompt into generic bullet lines.\n"
        "- Do not change the template_json; it is provided only for grounding.\n"
        "- Do not invent allowed values.\n"
        "- If the deterministic instructions already include exact current schema values or exact-match rules, preserve them.\n"
        "- Return JSON only: {\"instructions\": \"...\"}\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are editing extraction prompts for a schema-driven pipeline. "
                "Be precise, keep valid existing guidance, and only output JSON."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def _build_field_rewrite_messages(
    *,
    task_name: str,
    field_path: str,
    old_block: str,
    old_generated_rule: str,
    new_generated_rule: str,
    previous_block: str,
    next_block: str,
    action: str,
) -> List[Dict[str, str]]:
    payload = {
        "task_name": task_name,
        "field_path": field_path,
        "action": action,
        "old_field_block": old_block,
        "old_generated_rule": old_generated_rule,
        "new_generated_rule": new_generated_rule,
        "previous_field_block": previous_block,
        "next_field_block": next_block,
    }
    prompt = (
        "Update one field block inside an existing extraction prompt.\n"
        "Goals:\n"
        "- Preserve the original field-block style, detail, examples and edge-case rules whenever still valid.\n"
        "- Preserve existing lines verbatim unless they directly conflict with the new schema.\n"
        "- Change only what is required by the new schema.\n"
        "- Do not rewrite the whole task.\n"
        "- Do not collapse a rich field block into a generic one-line rule.\n"
        "- If action=replace: minimally edit the old block.\n"
        "- If action=add: write a new field block in the same style and detail level as neighboring blocks.\n"
        "- If action=remove: return action=remove and an empty block.\n"
        "- If the change is only an enum/value-list update, keep the old explanatory rules and only update the affected allowed-values and directly dependent rule lines.\n"
        "- Keep unaffected examples, caveats and rule bullets from the old block.\n"
        "- Use the exact current schema values from new_generated_rule when relevant.\n"
        "- Do not mention removed schema values in the final block.\n"
        "- Return JSON only: {\"action\":\"replace|add|remove\",\"block\":\"...\"}\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You edit one field block inside a detailed extraction prompt. "
                "Keep the old tone and structure, preserve valid examples, and only output JSON."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def _rewrite_task_field_blocks_with_llm(
    *,
    client: OpenAICompatibleClient,
    llm_cfg: Dict[str, Any],
    task_name: str,
    base_section: Dict[str, Any] | None,
    old_section: Dict[str, Any],
    new_section: Dict[str, Any],
    task_report: Dict[str, Any],
) -> Tuple[str | None, Dict[str, Any]]:
    if not base_section:
        return None, {}

    segments = _parse_instruction_segments(base_section.get("instructions"))
    if not any(seg.get("kind") == "field_block" for seg in segments):
        return None, {}

    old_lines = _field_lines(old_section.get("instructions"))
    new_lines = _field_lines(new_section.get("instructions"))
    diff = {
        "added_fields": list(task_report.get("added_fields") or []),
        "removed_fields": list(task_report.get("removed_fields") or []),
        "changed_fields": list(task_report.get("changed_fields") or []),
    }
    changed_fields = _changed_field_paths(old_section, new_section, diff)
    field_report: Dict[str, Any] = {}

    for field_path in changed_fields:
        idx = _find_segment_index_for_field(segments, field_path)
        old_block = ""
        previous_block = ""
        next_block = ""
        if idx >= 0:
            old_block = str(segments[idx].get("block") or "")
            for prev in reversed(segments[:idx]):
                if prev.get("kind") == "field_block":
                    previous_block = str(prev.get("block") or "")
                    break
            for nxt in segments[idx + 1:]:
                if nxt.get("kind") == "field_block":
                    next_block = str(nxt.get("block") or "")
                    break
        else:
            seen_current = False
            order = _new_field_order(new_section.get("instructions"))
            if field_path in order:
                for candidate in order:
                    seg_idx = _find_segment_index_for_field(segments, candidate)
                    if seg_idx < 0:
                        continue
                    if candidate == field_path:
                        seen_current = True
                        continue
                    if not seen_current:
                        previous_block = str(segments[seg_idx].get("block") or "")
                    elif not next_block:
                        next_block = str(segments[seg_idx].get("block") or "")
                        break

        action = "replace"
        if field_path in new_lines and field_path not in old_lines:
            action = "add"
        elif field_path in old_lines and field_path not in new_lines:
            action = "remove"

        messages = _build_field_rewrite_messages(
            task_name=task_name,
            field_path=field_path,
            old_block=old_block,
            old_generated_rule=old_lines.get(field_path, ""),
            new_generated_rule=new_lines.get(field_path, ""),
            previous_block=previous_block,
            next_block=next_block,
            action=action,
        )
        raw = client.chat(
            messages,
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=int(llm_cfg.get("max_tokens", 8000)),
            response_format={"type": "json_object"},
            timeout=int(llm_cfg.get("timeout", 900)),
        )
        payload = json.loads(raw)
        resolved_action = str(payload.get("action") or action).strip().lower() or action
        new_block = str(payload.get("block") or "").strip()
        field_report[field_path] = {
            "action": resolved_action,
            "has_old_block": bool(old_block),
        }

        if resolved_action == "remove":
            if idx >= 0:
                segments.pop(idx)
            continue

        if resolved_action == "replace" and idx >= 0 and new_block:
            segments[idx] = dict(segments[idx])
            segments[idx]["block"] = new_block
            continue

        if resolved_action == "add" and new_block:
            _insert_segment_for_field(
                segments,
                field_path=field_path,
                new_block=new_block,
                new_generated_section=new_section,
            )

    rewritten = _render_instruction_segments(segments)
    return (rewritten or None), field_report


def rewrite_changed_tasks_with_llm(
    *,
    updated_cfg: Dict[str, Dict[str, Any]],
    base_prompts: Dict[str, Dict[str, Any]],
    old_generated: Dict[str, Dict[str, Any]],
    new_generated: Dict[str, Dict[str, Any]],
    report: Dict[str, Any],
    llm_cfg: Dict[str, Any],
    llm_model: str | None = None,
) -> Dict[str, Any]:
    if not llm_cfg:
        raise RuntimeError("No [llm] config available for changed-task rewrite.")

    client = OpenAICompatibleClient(
        base_url=str(llm_cfg.get("base_url", "http://127.0.0.1:8080/v1")),
        api_key=str(llm_cfg.get("api_key", "sk-local")),
        model=str(llm_model or llm_cfg.get("model", "Qwen/Qwen2.5-32B-Instruct")),
        use_grammar=bool(llm_cfg.get("use_grammar", False)),
        use_session=bool(llm_cfg.get("sticky_session", False)),
    )

    rewrite_report: Dict[str, Any] = {}
    for task_name, task_report in sorted((report.get("tasks") or {}).items()):
        if not isinstance(task_report, dict) or not task_report.get("changed"):
            continue
        base_section = base_prompts.get(task_name)
        old_section = old_generated.get(task_name, {})
        new_section = updated_cfg.get(task_name) or new_generated.get(task_name, {})
        if not base_section:
            rewrite_report[task_name] = {
                "rewritten": False,
                "mode": "deterministic_only",
                "reason": "no_base_section",
            }
            continue
        field_rewrite, field_report = _rewrite_task_field_blocks_with_llm(
            client=client,
            llm_cfg=llm_cfg,
            task_name=task_name,
            base_section=base_section,
            old_section=old_section,
            new_section=new_generated.get(task_name, new_section),
            task_report=task_report,
        )
        if field_rewrite:
            updated_cfg[task_name] = dict(updated_cfg[task_name])
            updated_cfg[task_name]["instructions"] = field_rewrite
            rewrite_report[task_name] = {
                "rewritten": True,
                "mode": "field_blocks",
                "instruction_length": len(field_rewrite),
                "fields": field_report,
            }
            continue
        messages = _build_rewrite_messages(
            task_name=task_name,
            base_section=base_section,
            old_section=old_section,
            new_section=new_generated.get(task_name, new_section),
            diff={
                "added_fields": list(task_report.get("added_fields") or []),
                "removed_fields": list(task_report.get("removed_fields") or []),
                "changed_fields": list(task_report.get("changed_fields") or []),
            },
        )
        raw = client.chat(
            messages,
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=int(llm_cfg.get("max_tokens", 8000)),
            response_format={"type": "json_object"},
            timeout=int(llm_cfg.get("timeout", 900)),
        )
        payload = json.loads(raw)
        rewritten = str(payload.get("instructions") or "").strip()
        if not rewritten:
            raise RuntimeError(f"LLM returned empty instructions for {task_name}")
        updated_cfg[task_name] = dict(updated_cfg[task_name])
        updated_cfg[task_name]["instructions"] = rewritten
        rewrite_report[task_name] = {
            "rewritten": True,
            "mode": "full_task",
            "instruction_length": len(rewritten),
        }
    return rewrite_report


def build_prompt_comparison(
    *,
    base_prompts: Dict[str, Dict[str, Any]],
    old_generated: Dict[str, Dict[str, Any]],
    new_generated: Dict[str, Dict[str, Any]],
    final_cfg: Dict[str, Dict[str, Any]],
    report: Dict[str, Any],
    llm_report: Dict[str, Any],
) -> Dict[str, Any]:
    comparisons: Dict[str, Any] = {"tasks": {}}
    for task_name, task_report in sorted((report.get("tasks") or {}).items()):
        if not isinstance(task_report, dict) or not task_report.get("changed"):
            continue
        base_section = base_prompts.get(task_name) or {}
        old_section = old_generated.get(task_name) or {}
        new_section = new_generated.get(task_name) or {}
        final_section = final_cfg.get(task_name) or {}
        comparisons["tasks"][task_name] = {
            "changed": True,
            "instruction_changed": bool(task_report.get("instruction_changed")),
            "schema_diff": {
                "added_fields": list(task_report.get("added_fields") or []),
                "removed_fields": list(task_report.get("removed_fields") or []),
                "changed_fields": list(task_report.get("changed_fields") or []),
            },
            "llm_rewritten": bool((llm_report.get(task_name) or {}).get("rewritten")),
            "old_template_json": str(old_section.get("template_json") or ""),
            "new_template_json": str(new_section.get("template_json") or ""),
            "old_instructions": str(base_section.get("instructions") or ""),
            "deterministic_new_instructions": str(new_section.get("instructions") or ""),
            "final_instructions": str(final_section.get("instructions") or ""),
        }
    return comparisons


def _render_comparison_markdown(comparison: Dict[str, Any]) -> str:
    lines: List[str] = ["# Prompt Schema Update Comparison", ""]
    tasks = comparison.get("tasks") or {}
    if not tasks:
        lines.extend([
            "No changed tasks detected.",
            "",
        ])
        return "\n".join(lines)

    for task_name, payload in sorted(tasks.items()):
        if not isinstance(payload, dict):
            continue
        lines.append(f"## {task_name}")
        lines.append("")
        schema_diff = payload.get("schema_diff") or {}
        added = list(schema_diff.get("added_fields") or [])
        removed = list(schema_diff.get("removed_fields") or [])
        changed = list(schema_diff.get("changed_fields") or [])
        lines.append(f"- LLM rewritten: {'yes' if payload.get('llm_rewritten') else 'no'}")
        lines.append(f"- Instructions changed: {'yes' if payload.get('instruction_changed') else 'no'}")
        lines.append(f"- Added fields: {', '.join(added) if added else '(none)'}")
        lines.append(f"- Removed fields: {', '.join(removed) if removed else '(none)'}")
        lines.append(f"- Changed fields: {', '.join(changed) if changed else '(none)'}")
        lines.append("")
        lines.append("### Old Instructions")
        lines.append("")
        lines.append("```text")
        lines.append(str(payload.get("old_instructions") or ""))
        lines.append("```")
        lines.append("")
        lines.append("### Deterministic New Instructions")
        lines.append("")
        lines.append("```text")
        lines.append(str(payload.get("deterministic_new_instructions") or ""))
        lines.append("```")
        lines.append("")
        lines.append("### Final Instructions")
        lines.append("")
        lines.append("```text")
        lines.append(str(payload.get("final_instructions") or ""))
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _extract_field_block(instructions: str | None, field_path: str) -> str:
    segments = _parse_instruction_segments(instructions)
    idx = _find_segment_index_for_field(segments, field_path)
    if idx < 0:
        return ""
    block = str(segments[idx].get("block") or "").strip()
    return block


def _normalize_block_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _render_before_after_markdown(comparison: Dict[str, Any]) -> str:
    lines: List[str] = ["# Prompt Schema Before/After", ""]
    tasks = comparison.get("tasks") or {}
    if not tasks:
        lines.extend([
            "No changed tasks detected.",
            "",
        ])
        return "\n".join(lines)

    for task_name, payload in sorted(tasks.items()):
        if not isinstance(payload, dict):
            continue
        schema_diff = payload.get("schema_diff") or {}
        old_instructions = str(payload.get("old_instructions") or "")
        final_instructions = str(payload.get("final_instructions") or "")
        changed_fields = _changed_field_paths(
            {"instructions": old_instructions},
            {"instructions": final_instructions},
            schema_diff,
        )
        task_lines: List[str] = [f"## {task_name}", ""]
        rendered_any = False

        for field_path in changed_fields:
            before_block = _extract_field_block(old_instructions, field_path)
            after_block = _extract_field_block(final_instructions, field_path)
            if not before_block and not after_block:
                continue
            if _normalize_block_text(before_block) == _normalize_block_text(after_block):
                continue
            rendered_any = True
            task_lines.append(f"### `{field_path}`")
            task_lines.append("")
            task_lines.append("Before")
            task_lines.append("")
            task_lines.append("```text")
            task_lines.append(before_block or "(not present)")
            task_lines.append("```")
            task_lines.append("")
            task_lines.append("After")
            task_lines.append("")
            task_lines.append("```text")
            task_lines.append(after_block or "(not present)")
            task_lines.append("```")
            task_lines.append("")

        if not rendered_any:
            task_lines.append("No field-level prompt deltas could be isolated for this task.")
            task_lines.append("")

        lines.extend(task_lines)
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Update cohort prompts from an EMX2 schema diff while keeping unchanged prompt sections from the current TOML.")
    parser.add_argument("--base-prompts", required=True)
    parser.add_argument("--old-schema-csv", required=True)
    parser.add_argument("--new-schema-csv", required=True)
    parser.add_argument("--old-local-root", default=None)
    parser.add_argument("--new-local-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--profile", default="UMCGCohortsStaging")
    parser.add_argument("--rewrite-changed-with-llm", action="store_true")
    parser.add_argument("--llm-config", default="config.cohort.toml")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-report-json", default=None)
    parser.add_argument("--comparison-json", default=None)
    parser.add_argument("--comparison-md", default=None)
    parser.add_argument("--before-after-md", default=None)
    args = parser.parse_args()

    base_cfg = _task_sections(_load_toml(Path(args.base_prompts).expanduser().resolve()))
    old_registry = build_runtime_registry(
        args.profile,
        tables=None,
        local_root=args.old_local_root,
        fallback_schema_csv=args.old_schema_csv,
    )
    new_registry = build_runtime_registry(
        args.profile,
        tables=None,
        local_root=args.new_local_root,
        fallback_schema_csv=args.new_schema_csv,
    )
    old_generated = build_dynamic_task_sections(old_registry)
    new_generated = build_dynamic_task_sections(new_registry)

    updated_cfg, report = build_updated_prompt_cfg(
        base_prompts=base_cfg,
        old_generated=old_generated,
        new_generated=new_generated,
    )
    llm_report: Dict[str, Any] = {}
    if args.rewrite_changed_with_llm:
        llm_report = rewrite_changed_tasks_with_llm(
            updated_cfg=updated_cfg,
            base_prompts=base_cfg,
            old_generated=old_generated,
            new_generated=new_generated,
            report=report,
            llm_cfg=_load_llm_cfg(args.llm_config),
            llm_model=args.llm_model,
        )
    write_task_prompts_toml(updated_cfg, args.output)
    if args.report_json:
        write_json(args.report_json, report)
    if args.llm_report_json:
        write_json(args.llm_report_json, llm_report)
    comparison = build_prompt_comparison(
        base_prompts=base_cfg,
        old_generated=old_generated,
        new_generated=new_generated,
        final_cfg=updated_cfg,
        report=report,
        llm_report=llm_report,
    )
    if args.comparison_json:
        write_json(args.comparison_json, comparison)
    if args.comparison_md:
        out_path = Path(args.comparison_md).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_render_comparison_markdown(comparison), encoding="utf-8")
    if args.before_after_md:
        out_path = Path(args.before_after_md).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_render_before_after_markdown(comparison), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
