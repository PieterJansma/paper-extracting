from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml

from cohort_dynamic_prompts import build_dynamic_task_sections
from emx2_dynamic_runtime import COHORT_RUNTIME_TABLES, build_runtime_registry, write_json, write_task_prompts_toml
from llm_client import OpenAICompatibleClient


TASK_PREFIX = "task_"


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
        changed = any(diff.values())
        report["tasks"][task_name] = {
            "changed": changed,
            **diff,
        }

        if not changed and base_section is not None:
            out[task_name] = dict(base_section)
            continue
        out[task_name] = _prepend_schema_change_notice(task_name, new_section, diff)

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
    payload = {
        "task_name": task_name,
        "schema_diff": diff,
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
        "- Do not mention removed fields.\n"
        "- Do mention newly added fields when relevant.\n"
        "- Do not change the template_json; it is provided only for grounding.\n"
        "- Do not invent allowed values; those are injected dynamically elsewhere.\n"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Update cohort prompts from an EMX2 schema diff while keeping unchanged prompt sections from the current TOML.")
    parser.add_argument("--base-prompts", required=True)
    parser.add_argument("--old-schema-csv", required=True)
    parser.add_argument("--new-schema-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--profile", default="UMCGCohortsStaging")
    parser.add_argument("--rewrite-changed-with-llm", action="store_true")
    parser.add_argument("--llm-config", default="config.final.toml")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-report-json", default=None)
    parser.add_argument("--comparison-json", default=None)
    parser.add_argument("--comparison-md", default=None)
    args = parser.parse_args()

    base_cfg = _task_sections(_load_toml(Path(args.base_prompts).expanduser().resolve()))
    old_registry = build_runtime_registry(
        args.profile,
        tables=COHORT_RUNTIME_TABLES,
        fallback_schema_csv=args.old_schema_csv,
    )
    new_registry = build_runtime_registry(
        args.profile,
        tables=COHORT_RUNTIME_TABLES,
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
    print(args.output)


if __name__ == "__main__":
    main()
