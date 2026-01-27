from __future__ import annotations

import os
import logging
import json
import argparse
from typing import Any, Dict, List

import pandas as pd

try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml

from llm_client import OpenAICompatibleClient
from extract_pipeline import load_pdf_text, extract_fields, _merge_json_results

# ==============================================================================
# 1. DE OFFICIËLE VOLGORDE (Je Target Excel Headers)
# ==============================================================================
OFFICIAL_ORDER = [
    "rdf type", "fdp endpoint", "ldp membership relation", "hricore", "id",
    "pid", "name", "local name", "acronym", "type", "type other", "catalogue type",
    "cohort type", "clinical study type", "RWD type", "network type", "website",
    "description", "keywords", "internal identifiers.resource",
    "internal identifiers.identifier", "external identifiers.resource",
    "external identifiers.identifier", "start year", "end year",
    "time span description", "contact email", "logo", "logo_filename", "status",
    "conforms to", "has member relation", "issued", "modified", "design",
    "design description", "design schematic", "design schematic_filename",
    "data collection type", "data collection description", "reason sustained",
    "record trigger", "unit of observation", "subpopulations.resource",
    "subpopulations.name", "collection events.resource", "collection events.name",
    "data resources", "part of networks", "number of participants",
    "number of participants with samples", "underlying population",
    "population of interest", "population of interest other", "countries",
    "regions", "population age groups", "age min", "age max", "inclusion criteria",
    "other inclusion criteria", "exclusion criteria", "other exclusion criteria",
    "population entry", "population entry other", "population exit",
    "population exit other", "population disease", "population oncology topology",
    "population oncology morphology", "population coverage",
    "population not covered", "counts.resource", "counts.age group",
    "organisations involved.resource", "organisations involved.id",
    "publisher.resource", "publisher.id", "creator.resource", "creator.id",
    "people involved.resource", "people involved.first name",
    "people involved.last name", "contact point.resource",
    "contact point.first name", "contact point.last name", "child networks",
    "parent networks", "datasets.resource", "datasets.name", "samplesets.resource",
    "samplesets.name", "areas of information", "areas of information rwd",
    "quality of life other", "cause of death code other",
    "indication vocabulary other", "genetic data vocabulary other",
    "care setting other", "medicinal product vocabulary other",
    "prescriptions vocabulary other", "dispensings vocabulary other",
    "procedures vocabulary other", "biomarker data vocabulary other",
    "diagnosis medical event vocabulary other", "data dictionary available",
    "disease details", "biospecimen collected", "languages", "multiple entries",
    "has identifier", "identifier description", "prelinked", "linkage options",
    "linkage possibility", "linked resources.resource",
    "linked resources.linked resource", "informed consent type",
    "informed consent required", "informed consent other", "access rights",
    "data access conditions", "data use conditions",
    "data access conditions description", "data access fee",
    "access identifiable data", "access identifiable data route",
    "access subject details", "access subject details route", "access third party",
    "access third party conditions", "access non EU", "access non EU conditions",
    "biospecimen access", "biospecimen access conditions", "governance details",
    "approval for publication", "release type", "release description",
    "number of records", "release frequency", "refresh time", "lag time",
    "refresh period", "date last refresh", "preservation", "preservation duration",
    "standard operating procedures", "qualification", "qualifications description",
    "audit possible", "completeness", "completeness over time",
    "completeness results", "quality description", "quality over time",
    "access for validation", "quality validation frequency",
    "quality validation methods", "correction methods", "quality validation results",
    "mappings to common data models.source",
    "mappings to common data models.source dataset",
    "mappings to common data models.target",
    "mappings to common data models.target dataset", "common data models other",
    "ETL standard vocabularies", "ETL standard vocabularies other",
    "publications.resource", "publications.doi", "funding sources",
    "funding scheme", "funding statement", "citation requirements",
    "acknowledgements", "provenance statement", "documentation.resource",
    "documentation.name", "supplementary information", "theme",
    "applicable legislation", "collection start planned", "collection start actual",
    "analysis start planned", "analysis start actual", "data sources",
    "medical conditions studied", "data extraction date", "analysis plan",
    "objectives", "results", "mg_draft"
]

# ==============================================================================
# 2. KEY MAPPING (Van LLM Output -> Naar Excel Header)
# ==============================================================================
KEY_MAPPING = {
    # --- Pass A ---
    "pid": "pid",
    "study_name": "name",
    "local_name": "local name",
    "study_acronym": "acronym",
    "study_types": "type",
    "cohort_type": "cohort type",
    "study_status": "status",
    "website": "website",
    "start_year": "start year",
    "end_year": "end year",
    "contact_email": "contact email",
    "n_included": "number of participants",
    "countries": "countries",
    "regions": "regions",
    "population_age_group": "population age groups",
    "keywords": "keywords",

    # --- Pass B ---
    "inclusion_criteria": "inclusion criteria",
    "other_inclusion_criteria": "other inclusion criteria",
    "exclusion_criteria": "exclusion criteria",
    "other_exclusion_criteria": "other exclusion criteria",
    "clinical_study_types": "clinical study type",
    "type_other": "type other",
    "rwd_type": "RWD type",
    "network_type": "network type",

    # --- Pass C ---
    "design": "design",
    "design_description": "design description",
    "data_collection_type": "data collection type",
    "data_collection_description": "data collection description",
    "description": "description",
    "time_span_description": "time span description",

    # --- Pass D ---
    "number_of_participants_with_samples": "number of participants with samples",
    "underlying_population": "underlying population",
    "population_of_interest": "population of interest",
    "population_of_interest_other": "population of interest other",
    "part_of_networks": "part of networks",
    "population_entry": "population entry",
    "population_entry_other": "population entry other",
    "population_exit": "population exit",
    "population_exit_other": "population exit other",
    "population_disease": "population disease",
    "population_oncology_topology": "population oncology topology",
    "population_oncology_morphology": "population oncology morphology",
    "population_coverage": "population coverage",
    "population_not_covered": "population not covered",
    "age_min": "age min",
    "age_max": "age max",

    # --- Pass E ---
    "informed_consent_type": "informed consent type",
    "informed_consent_required": "informed consent required",
    "informed_consent_other": "informed consent other",
    "access_rights": "access rights",
    "data_access_conditions": "data access conditions",
    "data_use_conditions": "data use conditions",
    "data_access_conditions_description": "data access conditions description",
    "data_access_fee": "data access fee",
    "access_identifiable_data": "access identifiable data",
    "access_identifiable_data_route": "access identifiable data route",
    "access_subject_details": "access subject details",
    "access_subject_details_route": "access subject details route",
    "access_third_party": "access third party",
    "access_third_party_conditions": "access third party conditions",
    "access_non_eu": "access non EU",
    "access_non_eu_conditions": "access non EU conditions",

    # --- Pass F ---
    "counts_resource": "counts.resource",
    "counts_age_group": "counts.age group",
    "organisations_involved_resource": "organisations involved.resource",
    "organisations_involved_id": "organisations involved.id",
    "publisher_resource": "publisher.resource",
    "publisher_id": "publisher.id",
    "creator_resource": "creator.resource",
    "creator_id": "creator.id",
    "people_involved_resource": "people involved.resource",
    "contact_point_resource": "contact point.resource",
    "contact_point_first_name": "contact point.first name",
    "contact_point_last_name": "contact point.last name",
    "child_networks": "child networks",
    "parent_networks": "parent networks",

    # --- Pass G ---
    "datasets_resource": "datasets.resource",
    "datasets_name": "datasets.name",
    "samplesets_resource": "samplesets.resource",
    "samplesets_name": "samplesets.name",
    "areas_of_information": "areas of information",
    "areas_of_information_rwd": "areas of information rwd",
    "quality_of_life_measures": "quality of life other",
    "cause_of_death_vocabulary": "cause of death code other",
    "indication_vocabulary": "indication vocabulary other",
    "genetic_data_vocabulary": "genetic data vocabulary other",
    "care_setting_description": "care setting other",
    "medicinal_product_vocabulary": "medicinal product vocabulary other",
    "prescriptions_vocabulary": "prescriptions vocabulary other",
    "dispensings_vocabulary": "dispensings vocabulary other",
    "procedures_vocabulary": "procedures vocabulary other",
    "biomarker_data_vocabulary": "biomarker data vocabulary other",
    "diagnosis_medical_event_vocabulary": "diagnosis medical event vocabulary other",
    "data_dictionary_available": "data dictionary available",
    "disease_details": "disease details",

    # --- Pass H ---
    "biospecimen_access": "biospecimen access",
    "biospecimen_access_conditions": "biospecimen access conditions",
    "governance_details": "governance details",
    "approval_for_publication": "approval for publication",
    "release_type": "release type",
    "release_description": "release description",
    "number_of_records": "number of records",
    "release_frequency_months": "release frequency",
    "refresh_time_days": "refresh time",
    "lag_time_days": "lag time",
    "refresh_period": "refresh period",
    "date_last_refresh": "date last refresh",
    "preservation_indefinite": "preservation",
    "preservation_duration_years": "preservation duration",

    # --- Pass I ---
    "standard_operating_procedures": "standard operating procedures",
    "qualification": "qualification",
    "qualifications_description": "qualifications description",
    "audit_possible": "audit possible",
    "completeness": "completeness",
    "completeness_over_time": "completeness over time",
    "completeness_results": "completeness results",
    "quality_description": "quality description",
    "quality_over_time": "quality over time",
    "access_for_validation": "access for validation",
    "quality_validation_frequency": "quality validation frequency",
    "quality_validation_methods": "quality validation methods",
    "correction_methods": "correction methods",
    "quality_validation_results": "quality validation results",
    "quality_marks": "quality marks",

    # --- Pass J ---
    "biospecimen_collected": "biospecimen collected",
    "languages": "languages",
    "multiple_entries": "multiple entries",
    "has_identifier": "has identifier",
    "identifier_description": "identifier description",
    "prelinked": "prelinked",
    "linkage_options": "linkage options",
    "linkage_possibility": "linkage possibility",
    "linked_resources_names": "linked resources.resource",

    # --- Pass K ---
    "reason_sustained": "reason sustained",
    "record_trigger": "record trigger",
    "unit_of_observation": "unit of observation",
    "subpopulations_resource": "subpopulations.resource",
    "subpopulations_name": "subpopulations.name",
    "collection_events_resource": "collection events.resource",
    "collection_events_name": "collection events.name",
    "data_resources_included": "data resources",

    # --- Pass L ---
    "cdm_mapping_source": "mappings to common data models.source",
    "cdm_mapping_source_dataset": "mappings to common data models.source dataset",
    "cdm_mapping_target": "mappings to common data models.target",
    "cdm_mapping_target_dataset": "mappings to common data models.target dataset",
    "cdm_other": "common data models other",
    "etl_vocabularies": "ETL standard vocabularies",
    "etl_vocabularies_other": "ETL standard vocabularies other",
    "publications": "publications.resource",
    "funding_sources": "funding sources",
    "funding_scheme": "funding scheme",
    "funding_statement": "funding statement",

    # --- Pass M ---
    "citation_requirements": "citation requirements",
    "acknowledgements": "acknowledgements",
    "provenance_statement": "provenance statement",
    "documentation": "documentation.resource",
    "internal_identifiers": "internal identifiers.resource",
    "external_identifiers": "external identifiers.resource",
    "supplementary_information": "supplementary information",
    "theme": "theme",
    "applicable_legislation": "applicable legislation",
    "collection_start_planned": "collection start planned",
    "collection_start_actual": "collection start actual",
    "analysis_start_planned": "analysis start planned",
    "analysis_start_actual": "analysis start actual",
    "metadata_issued": "issued",
    "metadata_modified": "modified",

    # --- Pass N ---
    "data_sources": "data sources",
    "medical_conditions_studied": "medical conditions studied",
    "data_extraction_date": "data extraction date",
    "analysis_plan": "analysis plan",
    "objectives": "objectives",
    "results": "results",
}

# ==============================================================================
# Helpers
# ==============================================================================
def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return toml.load(f)


def transform_to_official_columns(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    transformed: Dict[str, Any] = {col: None for col in OFFICIAL_ORDER}
    for extraction_key, official_col in KEY_MAPPING.items():
        if extraction_key in extracted_data:
            val = extracted_data[extraction_key]
            if val is not None:
                transformed[official_col] = val
    return transformed


def _print_pretty_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def stem(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def merge_multi(combined: Dict[str, Any], label: str, res: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in res.items():
        if k not in combined or not isinstance(combined[k], dict):
            combined[k] = {}
        combined[k][label] = v
    return combined


def save_multi_to_excel(combined_raw: Dict[str, Any], labels: List[str], filename: str):
    """
    1 rij, maar kolommen zijn per paper gelabeld: "{label} :: {official_col}"
    """
    flat_row: Dict[str, Any] = {}

    for label in labels:
        per_paper = {
            k: (v.get(label) if isinstance(v, dict) else None)
            for k, v in combined_raw.items()
        }
        mapped = transform_to_official_columns(per_paper)

        for col in OFFICIAL_ORDER:
            out_col = f"{label} :: {col}"
            val = mapped.get(col)

            if val is None:
                flat_row[out_col] = ""
            elif isinstance(val, (list, dict)):
                flat_row[out_col] = json.dumps(val, ensure_ascii=False)
            else:
                flat_row[out_col] = str(val)

    df = pd.DataFrame([flat_row])
    df.to_excel(filename, index=False)
    logging.getLogger("main").info(f"Excel succesvol opgeslagen: {filename}")


# ==============================================================================
# CLI
# ==============================================================================
def cli():
    parser = argparse.ArgumentParser(description="Run PDF extraction passes.")
    parser.add_argument(
        "-p", "--passes",
        nargs="+",
        default=["all"],
        help="Specify passes (A, B, ... N) or 'all'."
    )
    parser.add_argument(
        "--pdfs",
        nargs="+",
        default=None,
        help="One or more PDF paths. Overrides config [pdf].path."
    )
    parser.add_argument(
        "--paper-names",
        nargs="+",
        default=None,
        help="Optional labels (same length as --pdfs). If omitted, uses filename stem."
    )
    parser.add_argument(
        "-o", "--output",
        default="final_result.xlsx",
        help="Naam van het output Excel bestand."
    )
    args = parser.parse_args()
    selected_passes = [p.upper() for p in args.passes]

    cfg_path = os.environ.get("PDF_EXTRACT_CONFIG", "config.toml")
    cfg = load_config(cfg_path)

    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    log = logging.getLogger("main")

    llm_cfg = cfg["llm"]
    pdf_cfg = cfg["pdf"]

    pdf_paths = args.pdfs if args.pdfs else [pdf_cfg["path"]]
    if args.paper_names and len(args.paper_names) != len(pdf_paths):
        raise SystemExit("--paper-names must have same length as --pdfs")

    labels = args.paper_names if args.paper_names else [stem(p) for p in pdf_paths]

    client = OpenAICompatibleClient(
        base_url=llm_cfg.get("base_url", "http://127.0.0.1:8080/v1"),
        api_key=llm_cfg.get("api_key", "sk-local"),
        model=llm_cfg.get("model", "numind/NuExtract-2.0-8B"),
        use_grammar=bool(llm_cfg.get("use_grammar", False)),
    )

    def run_pass(name: str, cfg_section_key: str, paper_text: str):
        task_cfg = cfg.get(cfg_section_key, {})
        if not task_cfg:
            log.warning(f"Sectie '{cfg_section_key}' is leeg of ontbreekt in config!")
            return {}

        log.info(f"--- Running {name} ---")
        return extract_fields(
            client, paper_text,
            template_json=task_cfg.get("template_json"),
            instructions=task_cfg.get("instructions"),
            use_grammar=bool(llm_cfg.get("use_grammar", False)),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=int(llm_cfg.get("max_tokens", 2048)),
        )

    all_tasks = [
        ("A", "Pass A (Main)", "task_main"),
        ("B", "Pass B (Criteria)", "task_criteria"),
        ("C", "Pass C (Design)", "task_design_details"),
        ("D", "Pass D (Population)", "task_population"),
        ("E", "Pass E (Access)", "task_access"),
        ("F", "Pass F (Contributors)", "task_contributors"),
        ("G", "Pass G (Data Model)", "task_datamodel"),
        ("H", "Pass H (Biobank Updates)", "task_biobank_updates"),
        ("I", "Pass I (Quality & SOPs)", "task_quality"),
        ("J", "Pass J (Linkage & Specs)", "task_linkage_specs"),
        ("K", "Pass K (Triggers & Structure)", "task_triggers_subpops"),
        ("L", "Pass L (CDM & Funding)", "task_cdm_funding"),
        ("M", "Pass M (Docs & Dates)", "task_docs_legislation_dates"),
        ("N", "Pass N (Content)", "task_study_content"),
    ]

    combined_results: Dict[str, Any] = {}

    for pdf_path, label in zip(pdf_paths, labels):
        paper_text = load_pdf_text(pdf_path, max_pages=pdf_cfg.get("max_pages"))
        log.info("PDF '%s' loaded (%d chars)", pdf_path, len(paper_text))

        per_paper_merged: Dict[str, Any] = {}

        for code, name, section in all_tasks:
            if "ALL" in selected_passes or code in selected_passes:
                res = run_pass(f"{name} [{code}] ({label})", section, paper_text)
                per_paper_merged = _merge_json_results(per_paper_merged, res)
            else:
                log.info(f"Skipping {name} (not selected)")

        combined_results = merge_multi(combined_results, label, per_paper_merged)

    log.info("--- DONE EXTRACTING ---")

    print("\n=== RAW EXTRACTED JSON (MULTI) ===")
    _print_pretty_json(combined_results)

    save_multi_to_excel(combined_results, labels, args.output)


if __name__ == "__main__":
    cli()
