from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml


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


def _task_sections(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(key): dict(value)
        for key, value in cfg.items()
        if str(key).startswith("task_") and isinstance(value, dict)
    }


def _as_lines(value: str | None) -> List[str]:
    return str(value or "").splitlines()


def _render_diff(old_text: str | None, new_text: str | None, *, from_name: str, to_name: str) -> str:
    diff = list(
        difflib.unified_diff(
            _as_lines(old_text),
            _as_lines(new_text),
            fromfile=from_name,
            tofile=to_name,
            lineterm="",
        )
    )
    if not diff:
        return "(no diff)"
    return "\n".join(diff)


def _render_schema_diff(payload: Dict[str, Any]) -> List[str]:
    schema_diff = payload.get("schema_diff") or {}
    added = list(schema_diff.get("added_fields") or [])
    removed = list(schema_diff.get("removed_fields") or [])
    changed = list(schema_diff.get("changed_fields") or [])
    lines = [
        f"- LLM rewritten: {'yes' if payload.get('llm_rewritten') else 'no'}",
        f"- Added fields: {', '.join(added) if added else '(none)'}",
        f"- Removed fields: {', '.join(removed) if removed else '(none)'}",
        f"- Changed fields: {', '.join(changed) if changed else '(none)'}",
    ]
    return lines


def build_summary_markdown(
    *,
    base_prompts: Dict[str, Dict[str, Any]],
    base_dynamic: Dict[str, Dict[str, Any]],
    variant_dynamic: Dict[str, Dict[str, Any]],
    updated_prompts: Dict[str, Dict[str, Any]],
    comparison: Dict[str, Any],
) -> str:
    tasks = comparison.get("tasks") or {}
    changed_task_names = [
        name for name, payload in sorted(tasks.items())
        if isinstance(payload, dict) and payload.get("changed")
    ]

    lines: List[str] = [
        "# Prompt Update Summary",
        "",
        "This file is meant for human review.",
        "It shows, per changed task:",
        "- how the prompt was before",
        "- what the schema-driven generator required to change",
        "- how the final prompt looks now",
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
        base_section = base_prompts.get(task_name) or {}
        base_dynamic_section = base_dynamic.get(task_name) or {}
        variant_dynamic_section = variant_dynamic.get(task_name) or {}
        updated_section = updated_prompts.get(task_name) or {}

        lines.append(f"## {task_name}")
        lines.append("")
        lines.extend(_render_schema_diff(payload))
        lines.append("")
        lines.append("### What The Schema Change Required")
        lines.append("")
        lines.append("#### Template Diff")
        lines.append("")
        lines.append("```diff")
        lines.append(
            _render_diff(
                base_dynamic_section.get("template_json"),
                variant_dynamic_section.get("template_json"),
                from_name="base_dynamic",
                to_name="variant_dynamic",
            )
        )
        lines.append("```")
        lines.append("")
        lines.append("#### Dynamic Prompt Diff")
        lines.append("")
        lines.append("```diff")
        lines.append(
            _render_diff(
                base_dynamic_section.get("instructions"),
                variant_dynamic_section.get("instructions"),
                from_name="base_dynamic",
                to_name="variant_dynamic",
            )
        )
        lines.append("```")
        lines.append("")
        lines.append("### Before")
        lines.append("")
        lines.append("```text")
        lines.append(str(base_section.get("instructions") or ""))
        lines.append("```")
        lines.append("")
        lines.append("### After")
        lines.append("")
        lines.append("```text")
        lines.append(str(updated_section.get("instructions") or ""))
        lines.append("```")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a compact human-friendly summary for prompt schema updates.")
    parser.add_argument("--base-prompts", required=True)
    parser.add_argument("--base-dynamic", required=True)
    parser.add_argument("--variant-dynamic", required=True)
    parser.add_argument("--updated-prompts", required=True)
    parser.add_argument("--comparison-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_prompts = _task_sections(_load_toml(args.base_prompts))
    base_dynamic = _task_sections(_load_toml(args.base_dynamic))
    variant_dynamic = _task_sections(_load_toml(args.variant_dynamic))
    updated_prompts = _task_sections(_load_toml(args.updated_prompts))
    comparison = _load_json(args.comparison_json)

    out = build_summary_markdown(
        base_prompts=base_prompts,
        base_dynamic=base_dynamic,
        variant_dynamic=variant_dynamic,
        updated_prompts=updated_prompts,
        comparison=comparison,
    )
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out, encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
