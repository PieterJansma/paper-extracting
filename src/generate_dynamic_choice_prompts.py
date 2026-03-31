from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml

import re

from emx2_dynamic_runtime import write_task_prompts_toml


TASK_PREFIX = "task_"
RUNTIME_NOTICE = (
    "DYNAMIC CHOICE RULES\n"
    "- Allowed values for choice/ref fields are injected at runtime from the current EMX2 model.\n"
    "- If a generated runtime constraints block is present, that block is authoritative.\n"
)


def _load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        data = toml.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid TOML root in {path}")
    return data


def _strip_manual_allowed_lists(text: str) -> str:
    pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)Allowed values(?: \(exact strings only\))?:\n"
        r"(?:(?P=indent)- .*\n)+"
    )
    return pattern.sub(
        lambda match: f"{match.group('indent')}Allowed values: injected dynamically at runtime.\n",
        text,
    )


def _transform_instructions(text: str) -> str:
    body = _strip_manual_allowed_lists(str(text or "").strip())
    if not body:
        return body
    if body.startswith("DYNAMIC CHOICE RULES"):
        return body
    return f"{RUNTIME_NOTICE}\n{body}".strip()


def build_dynamic_choice_prompts(source_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in source_cfg.items():
        if not str(key).startswith(TASK_PREFIX) or not isinstance(value, dict):
            continue
        section = dict(value)
        section["instructions"] = _transform_instructions(section.get("instructions", ""))
        out[key] = section
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a prompt TOML variant with dynamic choice lists.")
    parser.add_argument("input", help="Source prompts TOML")
    parser.add_argument("output", help="Output prompts TOML")
    args = parser.parse_args()

    source_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    cfg = _load_toml(source_path)
    transformed = build_dynamic_choice_prompts(cfg)
    write_task_prompts_toml(transformed, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
