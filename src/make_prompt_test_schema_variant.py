from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def _load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small EMX2 schema variant for prompt-update testing.")
    parser.add_argument("--input", default="schemas/molgenis_UMCGCohortsStaging.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    rows = _load_rows(in_path)
    if not rows:
        raise SystemExit(f"No rows in {in_path}")
    fieldnames = list(rows[0].keys())

    out_rows: List[Dict[str, str]] = []
    added = False
    removed = False

    for row in rows:
        table = str(row.get("tableName") or "")
        column = str(row.get("columnName") or "")

        if table == "Resources" and column == "access rights":
            removed = True
            continue

        out_rows.append(dict(row))

        if table == "Subpopulations" and column == "keywords" and not added:
            new_row = {key: "" for key in fieldnames}
            new_row.update({
                "tableName": "Subpopulations",
                "columnName": "recruitment channel",
                "columnType": "text",
                "key": "",
                "required": "",
                "refSchema": "",
                "refTable": "",
                "refLink": "",
                "refBack": "",
                "validation": "",
                "semantics": "",
                "description": "Recruitment channel for this subpopulation",
                "profiles": row.get("profiles", ""),
                "visible": row.get("visible", ""),
                "label": "Recruitment channel",
                "computed": "",
                "refLabel": "",
                "defaultValue": "",
            })
            out_rows.append(new_row)
            added = True

    if not removed:
        raise SystemExit("Could not remove test field Resources.access rights; source schema did not match expectation.")
    if not added:
        raise SystemExit("Could not insert test field Subpopulations.recruitment channel; source schema did not match expectation.")

    _write_rows(out_path, out_rows, fieldnames)
    print(out_path)


if __name__ == "__main__":
    main()
