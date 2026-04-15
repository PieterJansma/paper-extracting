from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_PROFILE = "UMCGCohortsStaging"
DEFAULT_SCHEMA_CSV = "schemas/molgenis_UMCGCohortsStaging.csv"

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
    "task_overview": {
        "internal_identifiers[]": "Internal identifiers",
        "external_identifiers[]": "External identifiers",
        "": "Resources",
    },
    "task_design_structure": {"": "Resources"},
    "task_subpopulations": {
        "subpopulations[].counts[]": "Subpopulation counts",
        "subpopulations[]": "Subpopulations",
    },
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
    "task_areas_of_information": {"": "Collection events"},
    "task_linkage": {"": "Resources"},
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def cohort_model_paths() -> List[str]:
    paths: List[str] = []
    seen: set[str] = set()
    for table_name in COHORT_RUNTIME_TABLES:
        rel_path = f"data/_models/shared/{table_name}.csv"
        if rel_path in seen:
            continue
        seen.add(rel_path)
        paths.append(rel_path)
    return paths


def profile_table_names(
    profile: str = DEFAULT_PROFILE,
    *,
    local_root: str | None = None,
    fallback_schema_csv: str | None = None,
    cache_dir: str | None = None,
) -> List[str]:
    rows, _ = load_profile_model_rows(
        profile,
        local_root=local_root,
        fallback_schema_csv=fallback_schema_csv,
        cache_dir=cache_dir,
    )
    names: List[str] = []
    seen: set[str] = set()
    for row in rows:
        table_name = str(row.get("tableName") or "").strip()
        if not table_name or table_name in seen:
            continue
        seen.add(table_name)
        names.append(table_name)
    return names


def profile_model_paths(
    profile: str = DEFAULT_PROFILE,
    *,
    local_root: str | None = None,
    fallback_schema_csv: str | None = None,
    cache_dir: str | None = None,
) -> List[str]:
    # Prefer the real shared model files that actually contain rows for this profile.
    # This avoids false lookups like `Collections.csv` when `tableName=Collections`
    # is defined inside another file (e.g. `Resources.csv`).
    for root in _candidate_local_roots(local_root):
        shared_dir = root / "data" / "_models" / "shared"
        if not shared_dir.is_dir():
            continue
        local_paths: List[str] = []
        for csv_path in sorted(shared_dir.glob("*.csv")):
            try:
                rows = _load_csv_rows(csv_path)
            except Exception:
                continue
            if any(profile in _split_profiles(row.get("profiles")) for row in rows):
                local_paths.append(f"data/_models/shared/{csv_path.name}")
        if local_paths:
            return local_paths

    # Fallback: derive paths from table names when local shared model files are unavailable.
    paths: List[str] = []
    seen: set[str] = set()
    for table_name in profile_table_names(
        profile,
        local_root=local_root,
        fallback_schema_csv=fallback_schema_csv,
        cache_dir=cache_dir,
    ):
        rel_path = f"data/_models/shared/{table_name}.csv"
        if rel_path in seen:
            continue
        seen.add(rel_path)
        paths.append(rel_path)
    return paths


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


def _candidate_cache_dirs(explicit_cache_dir: str | None = None) -> List[Path]:
    candidates = [
        explicit_cache_dir,
        os.environ.get("EMX2_CACHE_DIR"),
        str((_repo_root() / "tmp" / "emx2_runtime_cache").resolve()),
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


def _github_repo() -> str:
    return (os.environ.get("MOLGENIS_EMX2_REPO") or "molgenis/molgenis-emx2").strip() or "molgenis/molgenis-emx2"


def _github_refs() -> List[str]:
    refs = [
        (os.environ.get("MOLGENIS_EMX2_REF") or "main").strip(),
        "main",
        "master",
    ]
    out: List[str] = []
    seen: set[str] = set()
    for ref in refs:
        if not ref or ref in seen:
            continue
        seen.add(ref)
        out.append(ref)
    return out


def _http_get_bytes(url: str, timeout_sec: float = 20.0) -> bytes | None:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "paper-extracting-emx2-runtime",
        },
    )
    try:
        with urlopen(request, timeout=timeout_sec) as response:
            return response.read()
    except Exception:
        return None


def _github_shared_model_rel_paths(repo: str, ref: str) -> List[str]:
    api_url = f"https://api.github.com/repos/{repo}/contents/data/_models/shared?ref={quote(ref, safe='')}"
    payload = _http_get_bytes(api_url)
    if not payload:
        return []
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").strip() != "file":
            continue
        rel_path = str(item.get("path") or "").strip()
        if not rel_path.endswith(".csv"):
            continue
        if rel_path in seen:
            continue
        seen.add(rel_path)
        out.append(rel_path)
    return out


def _fetch_github_profile_rows(
    profile: str,
    *,
    cache_dir: str | None = None,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]] | None:
    repo = _github_repo()
    refs = _github_refs()
    cache_roots = _candidate_cache_dirs(cache_dir)

    for ref in refs:
        rel_paths = _github_shared_model_rel_paths(repo, ref)
        if not rel_paths:
            continue
        for cache_root in cache_roots:
            repo_root = (cache_root / "repo").resolve()
            rows: List[Dict[str, str]] = []
            fetched_files = 0
            for rel_path in rel_paths:
                raw_url = (
                    f"https://raw.githubusercontent.com/{repo}/{quote(ref, safe='')}/"
                    f"{quote(rel_path, safe='/')}"
                )
                payload = _http_get_bytes(raw_url)
                if not payload:
                    continue
                local_path = (repo_root / rel_path).resolve()
                try:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    local_path.write_bytes(payload)
                    fetched_files += 1
                except Exception:
                    continue

                try:
                    file_rows = _load_csv_rows(local_path)
                except Exception:
                    continue
                for row in file_rows:
                    if profile in _split_profiles(row.get("profiles")):
                        rows.append(row)
            if rows:
                return rows, {
                    "kind": "github_repo",
                    "repo": repo,
                    "ref": ref,
                    "cache_root": str(cache_root),
                    "repo_root": str(repo_root),
                    "shared_dir": str((repo_root / "data" / "_models" / "shared").resolve()),
                    "fetched_files": fetched_files,
                }
    return None


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
    cache_dir: str | None = None,
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

    github_result = _fetch_github_profile_rows(profile, cache_dir=cache_dir)
    if github_result is not None:
        return github_result

    schema_csv = _fallback_schema_csv(fallback_schema_csv)
    if schema_csv.is_file():
        rows = [
            row
            for row in _load_csv_rows(schema_csv)
            if profile in _split_profiles(row.get("profiles"))
        ]
        if rows:
            return rows, {
                "kind": "fallback_schema_csv",
                "path": str(schema_csv),
            }

    raise FileNotFoundError(
        "Could not resolve EMX2 profile model rows. Tried local shared models, "
        f"fallback schema CSV ({schema_csv}), and GitHub repo {_github_repo()} refs {_github_refs()}."
    )


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
        cache_dir=cache_dir,
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
                "description": str(row.get("description") or "").strip(),
                "label": str(row.get("label") or "").strip(),
                "semantics": str(row.get("semantics") or "").strip(),
                "validation": str(row.get("validation") or "").strip(),
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


def _resolve_task_table(path: str, target_map: Dict[str, str]) -> str:
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
        if len(values) <= 16:
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

    for task_name, task_cfg in (
        (str(name), value)
        for name, value in cfg.items()
        if str(name).startswith("task_") and isinstance(value, dict)
    ):
        target_map = COHORT_TASK_TEMPLATE_TARGETS.get(task_name, {})
        task_table = str(task_cfg.get("task_table") or "").strip()
        task_list_key = str(task_cfg.get("task_list_key") or "").strip()
        if task_table and task_list_key:
            target_map = {f"{task_list_key}[]": task_table}
        if not target_map:
            continue
        template = _parse_template_json(task_cfg.get("template_json"))
        if not template:
            continue

        lines: List[str] = []
        seen: set[str] = set()
        for path, leaf_name in _iter_template_paths(template):
            table_name = _resolve_task_table(path, target_map)
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


def export_profile_schema_csv(
    output_path: str | Path,
    *,
    profile: str = DEFAULT_PROFILE,
    local_root: str | None = None,
    fallback_schema_csv: str | None = None,
    cache_dir: str | None = None,
) -> Dict[str, Any]:
    rows, model_source = load_profile_model_rows(
        profile,
        local_root=local_root,
        fallback_schema_csv=fallback_schema_csv,
        cache_dir=cache_dir,
    )
    if not rows:
        raise RuntimeError(f"No profile rows found for {profile}")

    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return model_source


def write_task_prompts_toml(cfg: Dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    for section_name in sorted(k for k in cfg.keys() if str(k).startswith("task_")):
        section = cfg.get(section_name)
        if not isinstance(section, dict):
            continue
        lines.append(f"[{section_name}]")
        for key, value in section.items():
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                rendered = str(value)
            else:
                rendered = json.dumps("" if value is None else str(value), ensure_ascii=False)
            lines.append(f"{key} = {rendered}")
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build schema-driven EMX2 runtime metadata.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    req = sub.add_parser("required-paths", help="Print required EMX2 CSV paths for a mode/profile.")
    req.add_argument("--profile", default=DEFAULT_PROFILE)
    req.add_argument("--mode", default="cohort", choices=["cohort"])
    req.add_argument("--local-root", default=None)
    req.add_argument("--fallback-schema-csv", default=None)

    model = sub.add_parser("model-paths", help="Print shared model CSV paths needed for a mode/profile.")
    model.add_argument("--profile", default=DEFAULT_PROFILE)
    model.add_argument("--mode", default="cohort", choices=["cohort"])
    model.add_argument("--local-root", default=None)
    model.add_argument("--fallback-schema-csv", default=None)

    dump = sub.add_parser("dump-registry", help="Write resolved runtime registry as JSON.")
    dump.add_argument("--profile", default=DEFAULT_PROFILE)
    dump.add_argument("--mode", default="cohort", choices=["cohort"])
    dump.add_argument("--local-root", default=None)
    dump.add_argument("--fallback-schema-csv", default=None)
    dump.add_argument("--cache-dir", default=None)
    dump.add_argument("--output", required=True)

    export = sub.add_parser("export-schema-csv", help="Export a combined profile schema CSV from shared model files.")
    export.add_argument("--profile", default=DEFAULT_PROFILE)
    export.add_argument("--local-root", default=None)
    export.add_argument("--fallback-schema-csv", default=None)
    export.add_argument("--output", required=True)

    args = parser.parse_args()
    mode = getattr(args, "mode", "cohort")
    tables = COHORT_RUNTIME_TABLES if mode == "cohort" else ()

    if args.cmd == "required-paths":
        for rel_path in required_fetch_paths(
            args.profile,
            tables=tables,
            local_root=args.local_root,
            fallback_schema_csv=args.fallback_schema_csv,
        ):
            print(rel_path)
        return

    if args.cmd == "model-paths":
        if mode == "cohort":
            try:
                paths = profile_model_paths(
                    args.profile,
                    local_root=args.local_root,
                    fallback_schema_csv=args.fallback_schema_csv,
                )
            except Exception:
                paths = cohort_model_paths()
        else:
            paths = profile_model_paths(
                args.profile,
                local_root=args.local_root,
                fallback_schema_csv=args.fallback_schema_csv,
            )
        for rel_path in paths:
            print(rel_path)
        return

    if args.cmd == "export-schema-csv":
        model_source = export_profile_schema_csv(
            args.output,
            profile=args.profile,
            local_root=args.local_root,
            fallback_schema_csv=args.fallback_schema_csv,
        )
        print(json.dumps({"output": str(Path(args.output).resolve()), "model_source": model_source}, ensure_ascii=False))
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
