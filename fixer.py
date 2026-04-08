from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

INPUT_WORKBOOK_NAME = "alle_data_tabs_leesbaar.xlsx"
SELECTION_WORKBOOK_NAME = "selectie patienten.xlsx"
KEY_FILE_NAME = "sleutel_final(Sheet1).csv"

AZGNR_COLUMN = "AZGNR"
PATIENT_IDENTIFIER_COLUMN = "PatientIdentifier"
STUDY_ID_NEW_COLUMN = "studie_id_new"
STUDY_ID_COLUMN = "Study subject identifier"
SUBJECT_ID_COLUMN = "Subject identifier (technical)"
DOB_COLUMN = "DOB"
START_DATE_COLUMN = "startdt"
PRIMARY_SUBJECT_LOOKUP_SHEET = "data_NL_CM_0_1_1"

SELECTION_OUTPUT_SHEET = "selection_filtered"
SUMMARY_OUTPUT_SHEET = "filter_summary"


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def default_input_path() -> Path:
    return script_dir() / INPUT_WORKBOOK_NAME


def default_selection_path() -> Path:
    return script_dir() / SELECTION_WORKBOOK_NAME


def default_key_path() -> Path:
    return script_dir() / KEY_FILE_NAME


def default_output_path(input_path: Path, min_age: int, max_age: int) -> Path:
    return input_path.with_name(
        f"{input_path.stem}_selectie_leeftijd_{min_age}_{max_age}.xlsx"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map selectiepatienten via de sleutellijst naar studie_id_new, "
            "filter op leeftijd bij start behandeling, en pas die selectie toe "
            "op alle tabs van een Excel-werkmap."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input_path(),
        help="Bronbestand met alle leesbare data tabs (.xlsx).",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=default_selection_path(),
        help="Excelbestand met de patiëntselectie (.xlsx).",
    )
    parser.add_argument(
        "--selection-sheet",
        type=str,
        default=None,
        help="Optionele naam van het tabblad in het selectie-excelbestand.",
    )
    parser.add_argument(
        "--key",
        type=Path,
        default=default_key_path(),
        help="CSV met de mapping PatientIdentifier -> studie_id_new.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Gefilterde output-werkmap (.xlsx). "
            "Standaard: <input>_selectie_leeftijd_<min>_<max>.xlsx."
        ),
    )
    parser.add_argument("--min-age", type=int, default=18, help="Minimumleeftijd, inclusief.")
    parser.add_argument("--max-age", type=int, default=50, help="Maximumleeftijd, inclusief.")
    return parser.parse_args()


def normalize_value(value: object) -> str | None:
    if pd.isna(value):
        return None

    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value).strip() or None


def detect_identifier_width(series: pd.Series) -> int | None:
    lengths = series.map(normalize_value).dropna().map(len)
    if lengths.empty:
        return None
    return int(lengths.mode().iloc[0])


def normalize_identifier(value: object, width: int | None = None) -> str | None:
    normalized = normalize_value(value)
    if normalized is None:
        return None

    if normalized.endswith(".0") and normalized.replace(".0", "", 1).isdigit():
        normalized = normalized[:-2]

    if width is not None and normalized.isdigit():
        normalized = normalized.zfill(width)

    return normalized


def normalize_series(series: pd.Series, width: int | None = None) -> pd.Series:
    return series.map(lambda value: normalize_identifier(value, width=width))


def calculate_age_at_start(dob: pd.Series, start_date: pd.Series) -> pd.Series:
    before_birthday = (start_date.dt.month < dob.dt.month) | (
        (start_date.dt.month == dob.dt.month) & (start_date.dt.day < dob.dt.day)
    )
    return (
        start_date.dt.year
        - dob.dt.year
        - before_birthday.fillna(False).astype(int)
    )


def require_columns(frame: pd.DataFrame, required_columns: list[str], source_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise SystemExit(
            f"Missing column(s) in {source_name}: {', '.join(missing_columns)}"
        )


def read_selection_frame(path: Path, sheet_name: str | None) -> tuple[pd.DataFrame, str]:
    workbook = pd.ExcelFile(path)
    selected_sheet_name = sheet_name or workbook.sheet_names[0]
    if selected_sheet_name not in workbook.sheet_names:
        raise SystemExit(
            f"Selection sheet '{selected_sheet_name}' not found in {path.name}. "
            f"Available sheets: {', '.join(workbook.sheet_names)}"
        )

    frame = pd.read_excel(path, sheet_name=selected_sheet_name, dtype="object")
    require_columns(
        frame,
        [AZGNR_COLUMN, DOB_COLUMN, START_DATE_COLUMN],
        f"{path.name}:{selected_sheet_name}",
    )
    return frame, selected_sheet_name


def read_key_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype="object")
    require_columns(
        frame,
        [PATIENT_IDENTIFIER_COLUMN, STUDY_ID_NEW_COLUMN],
        path.name,
    )
    return frame


def prepare_selection(
    selection_frame: pd.DataFrame,
    key_frame: pd.DataFrame,
    min_age: int,
    max_age: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    patient_id_width = detect_identifier_width(key_frame[PATIENT_IDENTIFIER_COLUMN])

    key_map = key_frame[[PATIENT_IDENTIFIER_COLUMN, STUDY_ID_NEW_COLUMN]].copy()
    key_map["patient_identifier_normalized"] = normalize_series(
        key_map[PATIENT_IDENTIFIER_COLUMN], width=patient_id_width
    )
    key_map[STUDY_ID_NEW_COLUMN] = key_map[STUDY_ID_NEW_COLUMN].map(normalize_value)
    key_map = key_map.dropna(
        subset=["patient_identifier_normalized", STUDY_ID_NEW_COLUMN]
    ).copy()

    conflicts = (
        key_map.groupby("patient_identifier_normalized")[STUDY_ID_NEW_COLUMN]
        .nunique()
        .loc[lambda series: series > 1]
    )
    if not conflicts.empty:
        raise SystemExit(
            "Conflicting mappings found in key file for PatientIdentifier values: "
            + ", ".join(conflicts.index.tolist()[:10])
        )

    key_map = key_map.drop_duplicates(
        subset=["patient_identifier_normalized"], keep="first"
    )

    selection = selection_frame.copy()
    selection["azgnr_normalized"] = normalize_series(
        selection[AZGNR_COLUMN], width=patient_id_width
    )
    selection = selection.merge(
        key_map[["patient_identifier_normalized", STUDY_ID_NEW_COLUMN]],
        how="left",
        left_on="azgnr_normalized",
        right_on="patient_identifier_normalized",
    )

    selection[DOB_COLUMN] = pd.to_datetime(selection[DOB_COLUMN], errors="coerce")
    selection[START_DATE_COLUMN] = pd.to_datetime(
        selection[START_DATE_COLUMN], errors="coerce"
    )
    selection["age_at_start"] = calculate_age_at_start(
        selection[DOB_COLUMN], selection[START_DATE_COLUMN]
    )
    selection["has_study_id_new"] = selection[STUDY_ID_NEW_COLUMN].notna()
    selection["within_age_range"] = selection["age_at_start"].between(
        min_age, max_age, inclusive="both"
    )
    selection["selected_for_output"] = (
        selection["has_study_id_new"] & selection["within_age_range"]
    )

    filtered_selection = selection.loc[selection["selected_for_output"]].copy()
    return selection, filtered_selection, patient_id_width or 0


def collect_sheet_columns(path: Path, sheet_names: list[str]) -> dict[str, list[str]]:
    return {
        sheet_name: pd.read_excel(path, sheet_name=sheet_name, nrows=0).columns.tolist()
        for sheet_name in sheet_names
    }


def build_subject_lookup(
    path: Path,
    columns_by_sheet: dict[str, list[str]],
) -> pd.DataFrame:
    if PRIMARY_SUBJECT_LOOKUP_SHEET in columns_by_sheet:
        primary_columns = columns_by_sheet[PRIMARY_SUBJECT_LOOKUP_SHEET]
        if STUDY_ID_COLUMN in primary_columns and SUBJECT_ID_COLUMN in primary_columns:
            frame = pd.read_excel(
                path,
                sheet_name=PRIMARY_SUBJECT_LOOKUP_SHEET,
                usecols=[STUDY_ID_COLUMN, SUBJECT_ID_COLUMN],
                dtype="object",
            )
            frame[STUDY_ID_COLUMN] = normalize_series(frame[STUDY_ID_COLUMN])
            frame[SUBJECT_ID_COLUMN] = normalize_series(frame[SUBJECT_ID_COLUMN])
            return frame.dropna(
                subset=[STUDY_ID_COLUMN, SUBJECT_ID_COLUMN]
            ).drop_duplicates()

    lookup_frames: list[pd.DataFrame] = []
    for sheet_name, columns in columns_by_sheet.items():
        if STUDY_ID_COLUMN not in columns or SUBJECT_ID_COLUMN not in columns:
            continue

        frame = pd.read_excel(
            path,
            sheet_name=sheet_name,
            usecols=[STUDY_ID_COLUMN, SUBJECT_ID_COLUMN],
            dtype="object",
        )
        frame[STUDY_ID_COLUMN] = normalize_series(frame[STUDY_ID_COLUMN])
        frame[SUBJECT_ID_COLUMN] = normalize_series(frame[SUBJECT_ID_COLUMN])
        frame = frame.dropna(subset=[STUDY_ID_COLUMN, SUBJECT_ID_COLUMN]).drop_duplicates()
        if not frame.empty:
            lookup_frames.append(frame)

    if not lookup_frames:
        return pd.DataFrame(columns=[STUDY_ID_COLUMN, SUBJECT_ID_COLUMN])

    return pd.concat(lookup_frames, ignore_index=True).drop_duplicates()


def filter_sheet(
    frame: pd.DataFrame,
    eligible_study_ids: set[str],
    eligible_subject_ids: set[str],
) -> tuple[pd.DataFrame, str]:
    has_study_id = STUDY_ID_COLUMN in frame.columns
    has_subject_id = SUBJECT_ID_COLUMN in frame.columns

    if not has_study_id and not has_subject_id:
        return frame.copy(), "unfiltered"

    if has_study_id:
        study_mask = normalize_series(frame[STUDY_ID_COLUMN]).isin(eligible_study_ids)
    else:
        study_mask = pd.Series(False, index=frame.index)

    if has_subject_id:
        subject_mask = normalize_series(frame[SUBJECT_ID_COLUMN]).isin(eligible_subject_ids)
    else:
        subject_mask = pd.Series(False, index=frame.index)

    if has_study_id and has_subject_id:
        mask = study_mask | subject_mask
        filter_key = f"{STUDY_ID_COLUMN} or {SUBJECT_ID_COLUMN}"
    elif has_study_id:
        mask = study_mask
        filter_key = STUDY_ID_COLUMN
    else:
        mask = subject_mask
        filter_key = SUBJECT_ID_COLUMN

    return frame.loc[mask].copy(), filter_key


def main() -> None:
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    selection_path = args.selection.expanduser().resolve()
    key_path = args.key.expanduser().resolve()

    for path_label, path in (
        ("Input workbook", input_path),
        ("Selection workbook", selection_path),
        ("Key file", key_path),
    ):
        if not path.exists():
            raise SystemExit(f"{path_label} not found: {path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(input_path, args.min_age, args.max_age)
    )

    selection_frame, selection_sheet_name = read_selection_frame(
        selection_path, args.selection_sheet
    )
    key_frame = read_key_frame(key_path)
    selection_with_mapping, filtered_selection, patient_id_width = prepare_selection(
        selection_frame, key_frame, args.min_age, args.max_age
    )

    if filtered_selection.empty:
        raise SystemExit(
            "No patients remain after applying the key mapping and age filter."
        )

    eligible_study_ids = set(
        filtered_selection[STUDY_ID_NEW_COLUMN].dropna().astype(str).tolist()
    )

    workbook = pd.ExcelFile(input_path)
    columns_by_sheet = collect_sheet_columns(input_path, workbook.sheet_names)
    subject_lookup = build_subject_lookup(input_path, columns_by_sheet)
    eligible_subject_ids = set(
        subject_lookup.loc[
            subject_lookup[STUDY_ID_COLUMN].isin(eligible_study_ids), SUBJECT_ID_COLUMN
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    sheet_report: list[tuple[str, int, int, str]] = []
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name in workbook.sheet_names:
            print(f"Processing sheet: {sheet_name}", flush=True)
            frame = pd.read_excel(input_path, sheet_name=sheet_name, dtype="object")
            filtered_frame, filter_key = filter_sheet(
                frame, eligible_study_ids, eligible_subject_ids
            )
            filtered_frame.to_excel(writer, sheet_name=sheet_name, index=False)
            sheet_report.append((sheet_name, len(frame), len(filtered_frame), filter_key))

        selection_output = filtered_selection.copy()
        if patient_id_width > 0:
            selection_output[AZGNR_COLUMN] = normalize_series(
                selection_output[AZGNR_COLUMN], width=patient_id_width
            )
        selection_output = selection_output.drop(
            columns=["patient_identifier_normalized"], errors="ignore"
        )
        selection_output.to_excel(
            writer,
            sheet_name=SELECTION_OUTPUT_SHEET,
            index=False,
        )

        summary_frame = pd.DataFrame(
            sheet_report,
            columns=["sheet_name", "rows_input", "rows_kept", "filter_key"],
        )
        summary_frame.to_excel(writer, sheet_name=SUMMARY_OUTPUT_SHEET, index=False)

    print(f"Input workbook:             {input_path}")
    print(f"Selection workbook:         {selection_path} [{selection_sheet_name}]")
    print(f"Key file:                   {key_path}")
    print(f"Output workbook:            {output_path}")
    print(f"Selection rows before age:  {len(selection_with_mapping)}")
    print(
        "Selection rows with key:    "
        f"{int(selection_with_mapping['has_study_id_new'].sum())}"
    )
    print(f"Selection rows kept:        {len(filtered_selection)}")
    print(f"Eligible study ids:         {len(eligible_study_ids)}")
    print(f"Eligible subject ids:       {len(eligible_subject_ids)}")
    print("")
    print("Rows kept per sheet:")
    for sheet_name, total_rows, kept_rows, filter_key in sheet_report:
        print(f"- {sheet_name}: {kept_rows}/{total_rows} rows via {filter_key}")


if __name__ == "__main__":
    main()
