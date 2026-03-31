from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_PROFILE = "UMCGCohortsStaging"
DEFAULT_SCHEMA_CSV = "molgenis_UMCGCohortsStaging.csv"

COHORT_RUNTIME_TABLES: Tuple[str, ...] = (
    "Resources",
    "Subpopulations",
    "Subpopulation counts",
    "External identifiers",
    "Internal identifiers",
    "Collection events",
    "Agents",
    "Organisations",
    "Contacts",
    "Publications",
    "Documentation",
)

COHORT_TASK_TEMPLATE_TARGETS: Dict[str, Dict[str, str]] = {
    "task_overview": {"": "Resources"},
    "task_design_structure": {"": "Resources"},
    "task_subpopulations": {"subpopulations[]": "Subpopulations"},
    "task_collection_events": {"collection_events[]": "Collection events"},
    "task_collection_events_core": {"collection_events[]": "Collection events"},
    "task_collection_events_enrichment": {"collection_events[]": "Collection events"},
    "task_population": {"": "Resources"},
    "task_contributors": {
        "organisations_involved[]": "Organisations",
        "people_involved[]": "Contacts",
        "": "Resources",
    },
    "task_contributors_org": {
        "organisations_involved[]": "Organisations",
        "": "Resources",
    },
    "task_contributors_people": {
        "people_involved[]": "Contacts",
        "": "Resources",
    },
    "task_access_conditions": {"": "Resources"},
    "task_information": {
        "publications[]": "Publications",
        "documentation[]": "Documentation",
        "": "Resources",
    },
    "task_linkage": {"": "Resources"},
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _split_profiles(value: Any) -> set[str]:
    return {
        part.strip()
        for part in str(value or "").split(",")
        if part.strip()
    }


def _candidate_local_roots(explicit_root: str | None = None) -> List[Path]:
    candidates = [
        explicit_root,
        os.environ.get("MOLGENIS_EMX2_LOCAL_ROOT"),
        os.environ.get("EMX2_LOCAL_ROOT"),
    ]
    out: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _fallback_schema_csv(explicit_path: str | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    return (_repo_root() / DEFAULT_SCHEMA_CSV).resolve()


def _safe_cache_name(rel_path: str) -> str:
    return rel_path.replace("/", "__").replace(" ", "__")


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [
            {str(k or ""): str(v or "") for k, v in row.items()}
            for row in reader
        ]


def load_profile_model_rows(
    profile: str = DEFAULT_PROFILE,
    *,
    local_root: str | None = None,
    fallback_schema_csv: str | None = None,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    for root in _candidate_local_roots(local_root):
        shared_dir = root / "data" / "_models" / "shared"
        if not shared_dir.is_dir():
            continue
        rows: List[Dict[str, str]] = []
        for csv_path in sorted(shared_dir.glob("*.csv")):
            for row in _load_csv_rows(csv_path):
                if profile not in _split_profiles(row.get("profiles")):
                    continue
                rows.append(row)
        if rows:
            return rows, {
                "kind": "local_root",
                "root": str(root),
                "shared_dir": str(shared_dir),
            }

    schema_csv = _fallback_schema_csv(fallback_schema_csv)
    if not schema_csv.is_file():
        raise FileNotFoundError(f"Could not find fallback schema CSV: {schema_csv}")
    rows = [
        row
        for row in _load_csv_rows(schema_csv)
        if profile in _split_profiles(row.get("profiles"))
    ]
    return rows, {
        "kind": "fallback_schema_csv",
        "path": str(schema_csv),
    }


def _source_rel_paths_for_field(meta: Dict[str, Any]) -> List[str]:
    ref_schema = str(meta.get("ref_schema") or "").strip()
    ref_table = str(meta.get("ref_table") or "").strip()
    if not ref_table:
        return []
    if ref_schema == "CatalogueOntologies":
        return [f"data/_ontologies/{ref_table}.csv"]
    if ref_schema:
        return [
            f"data/_models/{ref_schema}/{ref_table}.csv",
            f"data/_models/shared/{ref_table}.csv",
            f"data/_ontologies/{ref_table}.csv",
        ]
    return []


def _resolve_source_path(
    rel_paths: Iterable[str],
    *,
    local_root: str | None = None,
    cache_dir: str | None = None,
) -> Tuple[Optional[str], Optional[str]]:
    roots = _candidate_local_roots(local_root)
    cwd = Path.cwd()
    cache_path = Path(cache_dir).expanduser().resolve() if cache_dir else None

    for rel_path in rel_paths:
        if not rel_path:
            continue
        for root in roots:
            candidate = root / rel_path
            if candidate.is_file():
                return rel_path, str(candidate.resolve())
        candidate = cwd / rel_path
        if candidate.is_file():
            return rel_path, str(candidate.resolve())
        if cache_path is not None:
            candidate = cache_path / _safe_cache_name(rel_path)
            if candidate.is_file():
                return rel_path, str(candidate.resolve())
    return None, None


def _load_choice_values(path: str | None) -> List[str]:
    if not path:
        return []
    csv_path = Path(path)
    if not csv_path.is_file():
        return []
    names: List[str] = []
    seen: set[str] = set()
    for row in _load_csv_rows(csv_path):
        name = str(row.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def build_runtime_registry(
    profile: str = DEFAULT_PROFILE,
    *,
    tables: Iterable[str] | None = None,
    local_root: str | None = None,
    fallback_schema_csv: str | None = None,
    cache_dir: str | None = None,
) -> Dict[str, Any]:
    rows, model_source = load_profile_model_rows(
        profile,
        local_root=local_root,
        fallback_schema_csv=fallback_schema_csv,
    )
    selected_tables = set(tables or [])

    table_extends: Dict[str, str] = {}
    direct_fields: Dict[str, List[Dict[str, Any]]] = {}
    all_tables: set[str] = set()

    for row in rows:
        table = str(row.get("tableName") or "").strip()
        if not table:
            continue
        all_tables.add(table)
        parent = str(row.get("tableExtends") or "").strip()
        if parent:
            table_extends[table] = parent
        column = str(row.get("columnName") or "").strip()
        if not column:
            continue
        if selected_tables and table not in selected_tables:
            continue
        direct_fields.setdefault(table, []).append(
            {
                "column_name": column,
                "column_type": str(row.get("columnType") or "").strip(),
                "ref_schema": str(row.get("refSchema") or "").strip(),
                "ref_table": str(row.get("refTable") or "").strip(),
                "ref_link": str(row.get("refLink") or "").strip(),
                "ref_back": str(row.get("refBack") or "").strip(),
                "required": str(row.get("required") or "").strip(),
                "default_value": str(row.get("defaultValue") or "").strip(),
            }
        )

    def resolve_table_fields(table_name: str, stack: Optional[set[str]] = None) -> Dict[str, Dict[str, Any]]:
        stack = stack or set()
        if table_name in stack:
            return {}
        stack.add(table_name)
        merged: Dict[str, Dict[str, Any]] = {}
        parent = table_extends.get(table_name)
        if parent and (not selected_tables or parent in selected_tables or parent in all_tables):
            merged.update(resolve_table_fields(parent, stack))
        for meta in direct_fields.get(table_name, []):
            merged[meta["column_name"]] = dict(meta)
        return merged

    tables_registry: Dict[str, Dict[str, Any]] = {}
    required_paths: List[str] = []
    sources: Dict[str, Dict[str, Any]] = {}

    for table_name in sorted(direct_fields.keys()):
        fields = resolve_table_fields(table_name)
        tables_registry[table_name] = {
            "extends": table_extends.get(table_name, ""),
            "fields": fields,
        }
        for meta in fields.values():
            rel_paths = _source_rel_paths_for_field(meta)
            if rel_paths:
                required_paths.extend(rel_paths)
            rel_path, source_path = _resolve_source_path(
                rel_paths,
                local_root=local_root,
                cache_dir=cache_dir,
            )
            if rel_path and rel_path not in sources:
                sources[rel_path] = {
                    "path": source_path or "",
                    "values": _load_choice_values(source_path),
                }
            if rel_path:
                meta["source_rel_path"] = rel_path
                meta["source_path"] = source_path or ""
                if rel_path in sources:
                    meta["allowed_values"] = list(sources[rel_path]["values"])

    deduped_required_paths: List[str] = []
    seen_paths: set[str] = set()
    for rel_path in required_paths:
        if not rel_path or rel_path in seen_paths:
            continue
        seen_paths.add(rel_path)
        deduped_required_paths.append(rel_path)

    return {
        "profile": profile,
        "model_source": model_source,
        "tables": tables_registry,
        "required_paths": deduped_required_paths,
        "sources": sources,
    }


def required_fetch_paths(
    profile: str = DEFAULT_PROFILE,
    *,
    tables: Iterable[str] | None = None,
    local_root: str | None = None,
    fallback_schema_csv: str | None = None,
) -> List[str]:
    registry = build_runtime_registry(
        profile,
        tables=tables,
        local_root=local_root,
        fallback_schema_csv=fallback_schema_csv,
        cache_dir=None,
    )
    return list(registry.get("required_paths", []))


def _parse_template_json(template_json: str | None) -> Dict[str, Any]:
    if not template_json:
        return {}
    try:
        return json.loads(template_json)
    except Exception:
        return {}


def _iter_template_paths(node: Any, prefix: str = "") -> Iterable[Tuple[str, str]]:
    if isinstance(node, dict):
        for key, value in node.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            if isinstance(value, list) and value and isinstance(value[0], dict):
                yield from _iter_template_paths(value[0], f"{new_prefix}[]")
            elif isinstance(value, dict):
                yield from _iter_template_paths(value, new_prefix)
            else:
                yield new_prefix, key


def _resolve_task_table(task_name: str, path: str) -> str:
    target_map = COHORT_TASK_TEMPLATE_TARGETS.get(task_name, {})
    best_prefix = ""
    best_table = ""
    for prefix, table in target_map.items():
        if not prefix and best_table:
            continue
        if prefix and not path.startswith(prefix):
            continue
        if len(prefix) >= len(best_prefix):
            best_prefix = prefix
            best_table = table
    if best_table:
        return best_table
    return target_map.get("", "")


def _lookup_field_meta(registry: Dict[str, Any], table_name: str, field_name: str) -> Dict[str, Any] | None:
    table_meta = registry.get("tables", {}).get(table_name, {})
    fields = table_meta.get("fields", {})
    wanted = _normalize_key(field_name)
    for column_name, meta in fields.items():
        if _normalize_key(column_name) == wanted:
            return meta
    return None


def _constraint_line(path: str, meta: Dict[str, Any]) -> str | None:
    column_type = meta.get("column_type")
    values = list(meta.get("allowed_values") or [])
    source_rel_path = str(meta.get("source_rel_path") or "").strip()
    ref_table = str(meta.get("ref_table") or "").strip()

    if column_type in {"ontology", "ontology_array"} and values:
        if len(values) <= 20:
            return f"- `{path}`: allowed values are {', '.join(values)}."
        return (
            f"- `{path}`: validated against current `{ref_table}.csv` "
            f"({len(values)} allowed values)."
        )

    if column_type in {"ref", "ref_array"} and source_rel_path:
        return (
            f"- `{path}`: validated against current `{ref_table}.csv` after extraction; "
            f"use the explicit value from the PDF."
        )

    return None


def build_dynamic_prompt_constraints(
    cfg: Dict[str, Any],
    registry: Dict[str, Any],
) -> Dict[str, List[str]]:
    summary: Dict[str, List[str]] = {}

    for task_name, target_map in COHORT_TASK_TEMPLATE_TARGETS.items():
        if task_name not in cfg or not target_map:
            continue
        template = _parse_template_json(cfg[task_name].get("template_json"))
        if not template:
            continue

        lines: List[str] = []
        seen: set[str] = set()
        for path, leaf_name in _iter_template_paths(template):
            table_name = _resolve_task_table(task_name, path)
            if not table_name:
                continue
            meta = _lookup_field_meta(registry, table_name, leaf_name)
            if not meta:
                continue
            line = _constraint_line(path, meta)
            if not line or line in seen:
                continue
            seen.add(line)
            lines.append(line)

        if lines:
            summary[task_name] = lines

    return summary


def apply_dynamic_constraints_to_config(
    cfg: Dict[str, Any],
    registry: Dict[str, Any],
) -> Dict[str, List[str]]:
    summary = build_dynamic_prompt_constraints(cfg, registry)
    if not summary:
        return summary

    header = (
        "AUTOMATIC EMX2 CONSTRAINTS\n"
        "- This block is generated from the current EMX2 model and ontology CSV files.\n"
        "- If this block conflicts with earlier manual allowed-value lists, this block is authoritative.\n"
    )

    for task_name, lines in summary.items():
        task_cfg = cfg.get(task_name)
        if not isinstance(task_cfg, dict):
            continue
        original = str(task_cfg.get("instructions") or "").rstrip()
        appendix = header + "\n" + "\n".join(lines)
        task_cfg["instructions"] = f"{original}\n\n{appendix}".strip()

    return summary


def write_json(path: str | Path, payload: Any) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build schema-driven EMX2 runtime metadata.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    req = sub.add_parser("required-paths", help="Print required EMX2 CSV paths for a mode/profile.")
    req.add_argument("--profile", default=DEFAULT_PROFILE)
    req.add_argument("--mode", default="cohort", choices=["cohort"])
    req.add_argument("--local-root", default=None)
    req.add_argument("--fallback-schema-csv", default=None)

    dump = sub.add_parser("dump-registry", help="Write resolved runtime registry as JSON.")
    dump.add_argument("--profile", default=DEFAULT_PROFILE)
    dump.add_argument("--mode", default="cohort", choices=["cohort"])
    dump.add_argument("--local-root", default=None)
    dump.add_argument("--fallback-schema-csv", default=None)
    dump.add_argument("--cache-dir", default=None)
    dump.add_argument("--output", required=True)

    args = parser.parse_args()
    tables = COHORT_RUNTIME_TABLES if args.mode == "cohort" else ()

    if args.cmd == "required-paths":
        for rel_path in required_fetch_paths(
            args.profile,
            tables=tables,
            local_root=args.local_root,
            fallback_schema_csv=args.fallback_schema_csv,
        ):
            print(rel_path)
        return

    registry = build_runtime_registry(
        args.profile,
        tables=tables,
        local_root=args.local_root,
        fallback_schema_csv=args.fallback_schema_csv,
        cache_dir=args.cache_dir,
    )
    write_json(args.output, registry)


if __name__ == "__main__":
    main()
