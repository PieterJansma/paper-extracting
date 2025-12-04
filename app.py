import streamlit as st
import os
import sys  # <--- NODIG
import tempfile
import json
import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import BytesIO

# ==============================================================================
# BELANGRIJK: Voeg de 'src' map toe aan het pad
# ==============================================================================
# Dit zorgt ervoor dat Python de bestanden in de map 'src' kan vinden
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Nu kunnen we gewoon importeren alsof de bestanden naast app.py staan
try:
    # Let op: Zorg dat in src/ bestand imports GEEN punt (.) hebben
    from llm_client import OpenAICompatibleClient
    from extract_pipeline import load_pdf_text, extract_fields, _merge_json_results
except ImportError as e:
    st.error(f"Fout bij importeren: {e}. Check of de bestanden in de map 'src' staan en geen relatieve imports (met punten) gebruiken.")
    st.stop()

# Importeer toml
try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml

# ==============================================================================
# 1. CONFIGURATIE & CONSTANTEN
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

KEY_MAPPING = {
    # --- Pass A ---
    "pid": "pid", "study_name": "name", "local_name": "local name",
    "study_acronym": "acronym", "study_types": "type", "cohort_type": "cohort type",
    "study_status": "status", "website": "website", "start_year": "start year",
    "end_year": "end year", "contact_email": "contact email", "n_included": "number of participants",
    "countries": "countries", "regions": "regions", "population_age_group": "population age groups",
    "keywords": "keywords",

    # --- Pass B ---
    "inclusion_criteria": "inclusion criteria", "other_inclusion_criteria": "other inclusion criteria",
    "exclusion_criteria": "exclusion criteria", "other_exclusion_criteria": "other exclusion criteria",
    "clinical_study_types": "clinical study type", "type_other": "type other",
    "rwd_type": "RWD type", "network_type": "network type",

    # --- Pass C ---
    "design": "design", "design_description": "design description",
    "data_collection_type": "data collection type", "data_collection_description": "data collection description",
    "description": "description", "time_span_description": "time span description",

    # --- Pass D ---
    "number_of_participants_with_samples": "number of participants with samples",
    "underlying_population": "underlying population", "population_of_interest": "population of interest",
    "population_of_interest_other": "population of interest other", "part_of_networks": "part of networks",
    "population_entry": "population entry", "population_entry_other": "population entry other",
    "population_exit": "population exit", "population_exit_other": "population exit other",
    "population_disease": "population disease", "population_oncology_topology": "population oncology topology",
    "population_oncology_morphology": "population oncology morphology", "population_coverage": "population coverage",
    "population_not_covered": "population not covered", "age_min": "age min", "age_max": "age max",

    # --- Pass E ---
    "informed_consent_type": "informed consent type", "informed_consent_required": "informed consent required",
    "informed_consent_other": "informed consent other", "access_rights": "access rights",
    "data_access_conditions": "data access conditions", "data_use_conditions": "data use conditions",
    "data_access_conditions_description": "data access conditions description", "data_access_fee": "data access fee",
    "access_identifiable_data": "access identifiable data", "access_identifiable_data_route": "access identifiable data route",
    "access_subject_details": "access subject details", "access_subject_details_route": "access subject details route",
    "access_third_party": "access third party", "access_third_party_conditions": "access third party conditions",
    "access_non_eu": "access non EU", "access_non_eu_conditions": "access non EU conditions",

    # --- Pass F ---
    "counts_resource": "counts.resource", "counts_age_group": "counts.age group",
    "organisations_involved_resource": "organisations involved.resource", "organisations_involved_id": "organisations involved.id",
    "publisher_resource": "publisher.resource", "publisher_id": "publisher.id",
    "creator_resource": "creator.resource", "creator_id": "creator.id",
    "people_involved_resource": "people involved.resource", "contact_point_resource": "contact point.resource",
    "contact_point_first_name": "contact point.first name", "contact_point_last_name": "contact point.last name",
    "child_networks": "child networks", "parent_networks": "parent networks",

    # --- Pass G ---
    "datasets_resource": "datasets.resource", "datasets_name": "datasets.name",
    "samplesets_resource": "samplesets.resource", "samplesets_name": "samplesets.name",
    "areas_of_information": "areas of information", "areas_of_information_rwd": "areas of information rwd",
    "quality_of_life_measures": "quality of life other", "cause_of_death_vocabulary": "cause of death code other",
    "indication_vocabulary": "indication vocabulary other", "genetic_data_vocabulary": "genetic data vocabulary other",
    "care_setting_description": "care setting other", "medicinal_product_vocabulary": "medicinal product vocabulary other",
    "prescriptions_vocabulary": "prescriptions vocabulary other", "dispensings_vocabulary": "dispensings vocabulary other",
    "procedures_vocabulary": "procedures vocabulary other", "biomarker_data_vocabulary": "biomarker data vocabulary other",
    "diagnosis_medical_event_vocabulary": "diagnosis medical event vocabulary other", "data_dictionary_available": "data dictionary available",
    "disease_details": "disease details",

    # --- Pass H ---
    "biospecimen_access": "biospecimen access", "biospecimen_access_conditions": "biospecimen access conditions",
    "governance_details": "governance details", "approval_for_publication": "approval for publication",
    "release_type": "release type", "release_description": "release description",
    "number_of_records": "number of records", "release_frequency_months": "release frequency",
    "refresh_time_days": "refresh time", "lag_time_days": "lag time",
    "refresh_period": "refresh period", "date_last_refresh": "date last refresh",
    "preservation_indefinite": "preservation", "preservation_duration_years": "preservation duration",

    # --- Pass I ---
    "standard_operating_procedures": "standard operating procedures", "qualification": "qualification",
    "qualifications_description": "qualifications description", "audit_possible": "audit possible",
    "completeness": "completeness", "completeness_over_time": "completeness over time",
    "completeness_results": "completeness results", "quality_description": "quality description",
    "quality_over_time": "quality over time", "access_for_validation": "access for validation",
    "quality_validation_frequency": "quality validation frequency", "quality_validation_methods": "quality validation methods",
    "correction_methods": "correction methods", "quality_validation_results": "quality validation results",
    "quality_marks": "quality marks",

    # --- Pass J ---
    "biospecimen_collected": "biospecimen collected", "languages": "languages",
    "multiple_entries": "multiple entries", "has_identifier": "has identifier",
    "identifier_description": "identifier description", "prelinked": "prelinked",
    "linkage_options": "linkage options", "linkage_possibility": "linkage possibility",
    "linked_resources_names": "linked resources.resource",

    # --- Pass K ---
    "reason_sustained": "reason sustained", "record_trigger": "record trigger",
    "unit_of_observation": "unit of observation", "subpopulations_resource": "subpopulations.resource",
    "subpopulations_name": "subpopulations.name", "collection_events_resource": "collection events.resource",
    "collection_events_name": "collection events.name", "data_resources_included": "data resources",

    # --- Pass L ---
    "cdm_mapping_source": "mappings to common data models.source",
    "cdm_mapping_source_dataset": "mappings to common data models.source dataset",
    "cdm_mapping_target": "mappings to common data models.target",
    "cdm_mapping_target_dataset": "mappings to common data models.target dataset",
    "cdm_other": "common data models other", "etl_vocabularies": "ETL standard vocabularies",
    "etl_vocabularies_other": "ETL standard vocabularies other", "publications": "publications.resource",
    "funding_sources": "funding sources", "funding_scheme": "funding scheme",
    "funding_statement": "funding statement",

    # --- Pass M ---
    "citation_requirements": "citation requirements", "acknowledgements": "acknowledgements",
    "provenance_statement": "provenance statement", "documentation": "documentation.resource",
    "internal_identifiers": "internal identifiers.resource", "external_identifiers": "external identifiers.resource",
    "supplementary_information": "supplementary information", "theme": "theme",
    "applicable_legislation": "applicable legislation", "collection_start_planned": "collection start planned",
    "collection_start_actual": "collection start actual", "analysis_start_planned": "analysis start planned",
    "analysis_start_actual": "analysis start actual", "metadata_issued": "issued", "metadata_modified": "modified",

    # --- Pass N ---
    "data_sources": "data sources", "medical_conditions_studied": "medical conditions studied",
    "data_extraction_date": "data extraction date", "analysis_plan": "analysis plan",
    "objectives": "objectives", "results": "results"
}

TASKS_INFO = {
    "A": "Pass A: Main Metadata (Name, PID, URL)",
    "B": "Pass B: Criteria & Types",
    "C": "Pass C: Design Details",
    "D": "Pass D: Population & Samples",
    "E": "Pass E: Access & Governance",
    "F": "Pass F: Contributors",
    "G": "Pass G: Data Model & Vocabularies",
    "H": "Pass H: Biobank Updates",
    "I": "Pass I: Quality & SOPs",
    "J": "Pass J: Linkage Specs",
    "K": "Pass K: Triggers & Structure",
    "L": "Pass L: CDM & Funding",
    "M": "Pass M: Docs, IDs & Dates",
    "N": "Pass N: Content & Results"
}

# ==============================================================================
# 2. HELPER FUNCTIES
# ==============================================================================

def load_app_config():
    """Laad config.toml."""
    cfg_path = os.environ.get("PDF_EXTRACT_CONFIG", "config.toml")
    with open(cfg_path, "rb") as f:
        return toml.load(f)

def scrape_site_for_images(url: str):
    """Probeert logo te scrapen van website."""
    if not url or not url.startswith("http"):
        return None
    
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # 1. Check Open Graph image (Facebook/LinkedIn plaatje)
        og_image = soup.find("meta", property="og:image")
        if og_image and og_image.get("content"):
            return og_image["content"]
            
        # 2. Check voor 'logo' in img src
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if 'logo' in src.lower():
                # Maak absolute URL als het relatief is
                if not src.startswith("http"):
                    return requests.compat.urljoin(url, src)
                return src
                
    except Exception as e:
        # We willen niet crashen als website niet werkt
        return None
    return None

def create_excel_bytes(raw_data: dict) -> BytesIO:
    """Maakt Excel bestand in geheugen (voor download knop)."""
    # 1. Transformeer
    transformed = {col: None for col in OFFICIAL_ORDER}
    for extraction_key, official_col in KEY_MAPPING.items():
        if extraction_key in raw_data:
            val = raw_data[extraction_key]
            if val is not None:
                transformed[official_col] = val

    # 2. Plat slaan
    flat_data = {}
    for k, v in transformed.items():
        if v is None:
            flat_data[k] = ""
        elif isinstance(v, (list, dict)):
            flat_data[k] = json.dumps(v, ensure_ascii=False)
        else:
            flat_data[k] = str(v)

    # 3. Naar Excel bytes
    df = pd.DataFrame([flat_data])
    df = df.reindex(columns=OFFICIAL_ORDER)
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Extractie')
    return output

# ==============================================================================
# 3. STREAMLIT UI
# ==============================================================================

st.set_page_config(page_title="PDF Biobank Extractor", layout="wide")

st.title("📄 PDF Biobank Extractor")
st.markdown("Upload een PDF, selecteer de extractie-taken en download het resultaat direct in Excel formaat.")

# --- SIDEBAR: Instellingen ---
with st.sidebar:
    st.header("Instellingen")
    
    # PDF Upload
    uploaded_file = st.file_uploader("Kies een PDF bestand", type=["pdf"])
    
    st.divider()
    
    # Pass Selectie
    st.subheader("Selecteer Taken")
    all_passes = st.checkbox("Alles Selecteren", value=True)
    
    selected_passes = []
    if all_passes:
        selected_passes = list(TASKS_INFO.keys())
        st.info("Alle 14 taken worden uitgevoerd.")
    else:
        for code, desc in TASKS_INFO.items():
            if st.checkbox(desc, value=False):
                selected_passes.append(code)
    
    st.divider()
    run_btn = st.button("🚀 Start Extractie", type="primary", disabled=(not uploaded_file or not selected_passes))

# --- MAIN: Uitvoering ---
# --- MAIN: Uitvoering ---
if run_btn and uploaded_file:
    # 1. Tijdelijk opslaan PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(uploaded_file.read())
        tmp_pdf_path = tmp_pdf.name

    # 2. Config laden
    cfg = load_app_config()
    llm_cfg = cfg["llm"]
    
    # 3. Client init
    client = OpenAICompatibleClient(
        base_url=llm_cfg.get("base_url"),
        api_key=llm_cfg.get("api_key"),
        model=llm_cfg.get("model"),
        use_grammar=bool(llm_cfg.get("use_grammar", False)),
    )

    # 4. PDF Tekst Laden
    with st.spinner("PDF Tekst aan het inlezen..."):
        paper_text = load_pdf_text(tmp_pdf_path, max_pages=cfg["pdf"].get("max_pages", 40))
    
    st.success(f"PDF geladen ({len(paper_text)} karakters). Extractie start nu...")
    st.divider()

    # --- LAYOUT SETUP VOOR LIVE RESULTATEN ---
    # We maken hier alvast de kolommen aan, zodat we ze tijdens het runnen kunnen vullen
    col_status, col_result = st.columns([1, 1])

    merged_results = {}

    # Rechts: Hier komt de JSON die steeds update. We maken een 'leeg vak' (placeholder)
    with col_result:
        st.subheader("📊 Live Resultaten")
        json_placeholder = st.empty() # Dit vakje kunnen we steeds overschrijven
        json_placeholder.info("Wachten op data...")

    # Links: De voortgang en status updates
    with col_status:
        st.subheader("⚙️ Voortgang")
        # We gebruiken st.status voor de logs
        with st.status("Extractie loopt...", expanded=True) as status:
            
            # Mapping van Code -> Config Sectie
            pass_mapping = {
                "A": "task_main", "B": "task_criteria", "C": "task_design_details",
                "D": "task_population", "E": "task_access", "F": "task_contributors",
                "G": "task_datamodel", "H": "task_biobank_updates", "I": "task_quality",
                "J": "task_linkage_specs", "K": "task_triggers_subpops", "L": "task_cdm_funding",
                "M": "task_docs_legislation_dates", "N": "task_study_content"
            }

            # 5. Loop door Passes
            for code in TASKS_INFO.keys(): # Volgorde behouden
                if code in selected_passes:
                    desc = TASKS_INFO[code]
                    section = pass_mapping[code]
                    
                    st.write(f"⏳ **{desc}**...")
                    
                    # Voer taak uit
                    task_cfg = cfg.get(section, {})
                    if task_cfg:
                        res = extract_fields(
                            client, paper_text,
                            template_json=task_cfg.get("template_json"),
                            instructions=task_cfg.get("instructions"),
                            use_grammar=bool(llm_cfg.get("use_grammar", False)),
                            temperature=0.0,
                            max_tokens=2048
                        )
                        # Resultaten samenvoegen
                        merged_results = _merge_json_results(merged_results, res)
                        
                        # --- HIER IS DE MAGIE: UPDATE DE JSON LIVE ---
                        json_placeholder.json(merged_results, expanded=False)
                        
                    else:
                        st.warning(f"Config sectie {section} ontbreekt!")
            
            # 6. Scraping (Optioneel)
            if "website" in merged_results and merged_results["website"]:
                st.write("🌐 Website gevonden. Zoeken naar logo...")
                logo_url = scrape_site_for_images(merged_results["website"])
                if logo_url:
                    merged_results["logo"] = logo_url
                    st.write(f"✅ Logo gevonden: {logo_url}")
                    # Laatste update van JSON met logo erbij
                    json_placeholder.json(merged_results, expanded=False)
                else:
                    st.write("❌ Geen logo gevonden.")

            status.update(label="Extractie Voltooid!", state="complete", expanded=False)

    # 7. Download Knop (verschijnt pas als alles klaar is)
    st.divider()
    st.success("✅ Alles klaar! Je kunt de Excel nu downloaden.")
    
    excel_data = create_excel_bytes(merged_results)
    
    st.download_button(
        label="📥 Download Excel (.xlsx)",
        data=excel_data.getvalue(),
        file_name=f"extractie_{uploaded_file.name.replace('.pdf', '')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary"
    )

    # Opruimen tmp file
    os.remove(tmp_pdf_path)

elif not uploaded_file:
    st.info("👈 Upload eerst een PDF in de zijbalk om te beginnen.")