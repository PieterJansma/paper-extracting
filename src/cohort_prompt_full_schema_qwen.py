from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from cohort_dynamic_prompts import build_dynamic_task_sections
from cohort_prompt_schema_updater import _load_llm_cfg, _load_toml, _task_sections
from emx2_dynamic_runtime import COHORT_RUNTIME_TABLES, build_runtime_registry, write_json, write_task_prompts_toml
from llm_client import OpenAICompatibleClient


def _build_messages(
    *,
    task_name: str,
    old_instructions: str,
    deterministic_instructions: str,
    template_json: str,
) -> List[Dict[str, str]]:
    payload = {
        "task_name": task_name,
        "old_instructions": old_instructions,
        "new_template_json": template_json,
        "deterministic_new_instructions": deterministic_instructions,
    }
    prompt = (
        "Rewrite the prompt instructions for one extraction task.\n"
        "Goals:\n"
        "- Keep the useful style, nuance and practical extraction guidance from the old instructions when still valid.\n"
        "- Align exactly with the deterministic schema-driven instructions.\n"
        "- Do not invent allowed values.\n"
        "- If the deterministic instructions already include exact current schema values or exact-match rules, preserve them.\n"
        "- Do not change the template_json; it is provided only for grounding.\n"
        "- Keep the instructions concise and operational for an extraction model.\n"
        "- Return JSON only: {\"instructions\": \"...\"}\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are editing extraction prompts for a schema-driven pipeline. "
                "Preserve valid existing guidance, improve clarity, and only output JSON."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def _comparison_markdown(tasks: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = ["# Full Schema Qwen Prompt Comparison", ""]
    for task_name, payload in sorted(tasks.items()):
        lines.append(f"## {task_name}")
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
    parser = argparse.ArgumentParser(description="Generate a full cohort prompt from live EMX2 CSV files and rewrite all task instructions with Qwen.")
    parser.add_argument("--base-prompts", default="prompts/prompts_cohort.toml")
    parser.add_argument("--profile", default="UMCGCohortsStaging")
    parser.add_argument("--local-root", default=None)
    parser.add_argument("--fallback-schema-csv", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--llm-config", default="config.final.toml")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--comparison-json", default=None)
    parser.add_argument("--comparison-md", default=None)
    parser.add_argument("--llm-report-json", default=None)
    args = parser.parse_args()

    base_cfg = _task_sections(_load_toml(Path(args.base_prompts).expanduser().resolve()))
    registry = build_runtime_registry(
        args.profile,
        tables=COHORT_RUNTIME_TABLES,
        local_root=args.local_root,
        fallback_schema_csv=args.fallback_schema_csv,
        cache_dir=args.cache_dir,
    )
    deterministic_cfg = build_dynamic_task_sections(registry)
    final_cfg: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in deterministic_cfg.items()}

    llm_cfg = _load_llm_cfg(args.llm_config)
    if not llm_cfg:
        raise RuntimeError("No [llm] config available for full Qwen rewrite.")

    client = OpenAICompatibleClient(
        base_url=str(llm_cfg.get("base_url", "http://127.0.0.1:8080/v1")),
        api_key=str(llm_cfg.get("api_key", "sk-local")),
        model=str(args.llm_model or llm_cfg.get("model", "Qwen/Qwen2.5-32B-Instruct")),
        use_grammar=bool(llm_cfg.get("use_grammar", False)),
        use_session=bool(llm_cfg.get("sticky_session", False)),
    )

    llm_report: Dict[str, Any] = {}
    comparison: Dict[str, Any] = {"tasks": {}}
    for task_name, section in sorted(deterministic_cfg.items()):
        base_section = base_cfg.get(task_name) or {}
        messages = _build_messages(
            task_name=task_name,
            old_instructions=str(base_section.get("instructions") or ""),
            deterministic_instructions=str(section.get("instructions") or ""),
            template_json=str(section.get("template_json") or ""),
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

        final_cfg[task_name] = dict(section)
        final_cfg[task_name]["instructions"] = rewritten
        llm_report[task_name] = {
            "rewritten": True,
            "instruction_length": len(rewritten),
        }
        comparison["tasks"][task_name] = {
            "old_instructions": str(base_section.get("instructions") or ""),
            "deterministic_new_instructions": str(section.get("instructions") or ""),
            "final_instructions": rewritten,
            "template_json": str(section.get("template_json") or ""),
        }

    write_task_prompts_toml(final_cfg, args.output)
    if args.llm_report_json:
        write_json(args.llm_report_json, llm_report)
    if args.comparison_json:
        write_json(args.comparison_json, comparison)
    if args.comparison_md:
        out_path = Path(args.comparison_md).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_comparison_markdown(comparison.get("tasks") or {}), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
