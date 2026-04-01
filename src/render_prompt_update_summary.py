from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml


FIELD_LINE_RE = re.compile(r"^\s*-\s+`([^`]+)`:\s*(.*)$")
SOURCE_RE = re.compile(r"Source:\s*([^.]+)\.(.+?)\s*$")


def _load_toml(path: str | Path) -> Dict[str, Any]:
    with Path(path).expanduser().resolve().open("rb") as f:
        data = toml.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid TOML root in {path}")
    return data


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid JSON root in {path}")
    return data


def _load_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [{str(k or ""): str(v or "") for k, v in row.items()} for row in reader]


def _task_sections(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(key): dict(value)
        for key, value in cfg.items()
        if str(key).startswith("task_") and isinstance(value, dict)
    }


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _field_lines(section: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw_line in str(section.get("instructions") or "").splitlines():
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


def _source_ref(line: str | None) -> Tuple[str, str]:
    text = str(line or "")
    match = SOURCE_RE.search(text)
    if not match:
        return "", ""
    return str(match.group(1)).strip(), str(match.group(2)).strip()


def _schema_row_map(path: str | Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in _load_csv_rows(path):
        table = str(row.get("tableName") or "").strip()
        column = str(row.get("columnName") or "").strip()
        if not table or not column:
            continue
        out[(table, column)] = row
    return out


def _load_ontology_values(local_root: str | Path | None, ref_table: str) -> List[str]:
    if not local_root or not ref_table:
        return []
    path = Path(local_root).expanduser().resolve() / "data" / "_ontologies" / f"{ref_table}.csv"
    if not path.is_file():
        return []
    values: List[str] = []
    seen: set[str] = set()
    for row in _load_csv_rows(path):
        name = str(row.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        values.append(name)
    return values


def _row_summary(row: Dict[str, str] | None) -> str:
    if not row:
        return "(no schema row)"
    pieces: List[str] = []
    column_type = str(row.get("columnType") or "").strip()
    ref_table = str(row.get("refTable") or "").strip()
    description = str(row.get("description") or "").strip()
    if column_type:
        pieces.append(f"type=`{column_type}`")
    if ref_table:
        pieces.append(f"refTable=`{ref_table}`")
    if description:
        pieces.append(f"description={description}")
    return "; ".join(pieces) if pieces else "(empty schema row)"


def _markdown_escape(text: str) -> str:
    return str(text or "").replace("|", "\\|")


def _field_sort_key(path: str) -> Tuple[int, str]:
    return (path.count("."), path)


def _resolve_display_field(path: str) -> str:
    return path.strip() or "(unknown)"


def _find_best_original_block(instructions: str | None, field_path: str) -> str:
    blocks = _extract_original_field_blocks(instructions)
    leaf = str(field_path).split(".")[-1].replace("[]", "")
    return blocks.get(_normalize(leaf), "")


def _changed_field_paths(
    task_report: Dict[str, Any],
    base_lines: Dict[str, str],
    variant_lines: Dict[str, str],
) -> List[str]:
    changed: set[str] = set()

    for key in set(base_lines) | set(variant_lines):
        if base_lines.get(key) != variant_lines.get(key):
            changed.add(key)

    line_keys = set(base_lines) | set(variant_lines)

    def resolve_leaf(leaf: str) -> str:
        if leaf in line_keys:
            return leaf
        for key in sorted(line_keys):
            if key.endswith(f".{leaf}") or key.endswith(f"[]{'.' + leaf}"):
                return key
        return leaf

    schema_diff = task_report.get("schema_diff") or {}
    for leaf in list(schema_diff.get("added_fields") or []):
        changed.add(resolve_leaf(str(leaf)))
    for leaf in list(schema_diff.get("removed_fields") or []):
        changed.add(resolve_leaf(str(leaf)))
    for leaf in list(schema_diff.get("changed_fields") or []):
        changed.add(resolve_leaf(str(leaf)))

    return sorted(changed, key=_field_sort_key)


def _csv_change_lines(
    *,
    field_path: str,
    before_line: str,
    after_line: str,
    base_schema: Dict[Tuple[str, str], Dict[str, str]],
    variant_schema: Dict[Tuple[str, str], Dict[str, str]],
    base_local_root: str | None,
    variant_local_root: str | None,
) -> List[str]:
    table, column = _source_ref(after_line or before_line)
    if not table or not column:
        return ["- CSV source: could not resolve from prompt line."]

    base_row = base_schema.get((table, column))
    variant_row = variant_schema.get((table, column))
    lines: List[str] = []
    model_csv = f"`{table}.csv`"

    if base_row and not variant_row:
        lines.append(f"- CSV change: field removed from {model_csv}.")
        lines.append(f"- Old row: {_row_summary(base_row)}")
        return lines

    if variant_row and not base_row:
        lines.append(f"- CSV change: field added to {model_csv}.")
        lines.append(f"- New row: {_row_summary(variant_row)}")
        return lines

    if base_row and variant_row:
        if _row_summary(base_row) != _row_summary(variant_row):
            lines.append(f"- CSV change: schema row changed in {model_csv}.")
            lines.append(f"- Before row: {_row_summary(base_row)}")
            lines.append(f"- After row: {_row_summary(variant_row)}")

        ref_table = str((variant_row or base_row).get("refTable") or "").strip()
        column_type = str((variant_row or base_row).get("columnType") or "").strip()
        if ref_table and column_type in {"ontology", "ontology_array"}:
            base_values = _load_ontology_values(base_local_root, ref_table)
            variant_values = _load_ontology_values(variant_local_root, ref_table)
            added = [v for v in variant_values if v not in base_values]
            removed = [v for v in base_values if v not in variant_values]
            if added or removed:
                lines.append(f"- CSV change: `{ref_table}.csv` changed.")
                if added:
                    lines.append(f"- Added allowed values: {', '.join(added)}")
                if removed:
                    lines.append(f"- Removed allowed values: {', '.join(removed)}")

    if not lines:
        lines.append("- CSV change: no row-level difference found; change comes from prompt wording only.")
    return lines


def build_summary_markdown(
    *,
    base_prompts: Dict[str, Dict[str, Any]],
    base_dynamic: Dict[str, Dict[str, Any]],
    variant_dynamic: Dict[str, Dict[str, Any]],
    updated_prompts: Dict[str, Dict[str, Any]],
    comparison: Dict[str, Any],
    base_schema: Dict[Tuple[str, str], Dict[str, str]],
    variant_schema: Dict[Tuple[str, str], Dict[str, str]],
    base_local_root: str | None,
    variant_local_root: str | None,
) -> str:
    tasks = comparison.get("tasks") or {}
    changed_task_names = [
        name for name, payload in sorted(tasks.items())
        if isinstance(payload, dict) and payload.get("changed")
    ]

    lines: List[str] = [
        "# Prompt Update Summary",
        "",
        "This file is field-level and human-readable.",
        "For every changed field it shows:",
        "- what changed in the source CSVs",
        "- the prompt rule before",
        "- the prompt rule that had to change",
        "- the final prompt rule now",
        "",
        f"Changed tasks: {', '.join(changed_task_names) if changed_task_names else '(none)'}",
        "",
    ]

    if not changed_task_names:
        lines.append("No changed tasks detected.")
        lines.append("")
        return "\n".join(lines)

    for task_name in changed_task_names:
        payload = tasks[task_name]
        base_prompt_section = base_prompts.get(task_name) or {}
        base_dynamic_section = base_dynamic.get(task_name) or {}
        variant_dynamic_section = variant_dynamic.get(task_name) or {}
        updated_section = updated_prompts.get(task_name) or {}

        base_lines = _field_lines(base_dynamic_section)
        variant_lines = _field_lines(variant_dynamic_section)
        updated_lines = _field_lines(updated_section)
        field_paths = _changed_field_paths(payload, base_lines, variant_lines)

        lines.append(f"## {task_name}")
        lines.append("")
        lines.append(f"- LLM rewritten: {'yes' if payload.get('llm_rewritten') else 'no'}")
        lines.append("")

        for field_path in field_paths:
            before_line = base_lines.get(field_path, "")
            needed_line = variant_lines.get(field_path, "")
            final_line = updated_lines.get(field_path, needed_line)
            before_block = _find_best_original_block(base_prompt_section.get("instructions"), field_path)
            final_block = _find_best_original_block(updated_section.get("instructions"), field_path)

            lines.append(f"### `{_resolve_display_field(field_path)}`")
            lines.append("")
            lines.extend(
                _csv_change_lines(
                    field_path=field_path,
                    before_line=before_line,
                    after_line=needed_line or final_line,
                    base_schema=base_schema,
                    variant_schema=variant_schema,
                    base_local_root=base_local_root,
                    variant_local_root=variant_local_root,
                )
            )
            lines.append("")
            lines.append("Before prompt:")
            lines.append("```text")
            lines.append(before_block or before_line or "(field not present)")
            lines.append("```")
            lines.append("")
            lines.append("Needed after schema change:")
            lines.append("```text")
            lines.append(needed_line or "(field removed from generated prompt)")
            lines.append("```")
            lines.append("")
            lines.append("Final prompt now:")
            lines.append("```text")
            lines.append(final_block or final_line or "(field removed from final prompt)")
            lines.append("```")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a compact field-level summary for prompt schema updates.")
    parser.add_argument("--base-prompts", required=True)
    parser.add_argument("--base-dynamic", required=True)
    parser.add_argument("--variant-dynamic", required=True)
    parser.add_argument("--updated-prompts", required=True)
    parser.add_argument("--comparison-json", required=True)
    parser.add_argument("--base-schema-csv", required=True)
    parser.add_argument("--variant-schema-csv", required=True)
    parser.add_argument("--base-local-root", default=None)
    parser.add_argument("--variant-local-root", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_prompts = _task_sections(_load_toml(args.base_prompts))
    base_dynamic = _task_sections(_load_toml(args.base_dynamic))
    variant_dynamic = _task_sections(_load_toml(args.variant_dynamic))
    updated_prompts = _task_sections(_load_toml(args.updated_prompts))
    comparison = _load_json(args.comparison_json)
    base_schema = _schema_row_map(args.base_schema_csv)
    variant_schema = _schema_row_map(args.variant_schema_csv)

    out = build_summary_markdown(
        base_dynamic=base_dynamic,
        base_prompts=base_prompts,
        variant_dynamic=variant_dynamic,
        updated_prompts=updated_prompts,
        comparison=comparison,
        base_schema=base_schema,
        variant_schema=variant_schema,
        base_local_root=args.base_local_root,
        variant_local_root=args.variant_local_root,
    )
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out, encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
