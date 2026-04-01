# Prompt Schema Update Comparison

## task_access_conditions

- LLM rewritten: no
- Added fields: (none)
- Removed fields: access_rights
- Changed fields: (none)

### Old Instructions

```text
Return ONLY valid JSON matching the template. No extra text. Ensure all quotes are closed.

Extract ONLY RESOURCE-LEVEL access conditions (UI chapter: access conditions). Do NOT infer.
Use ONLY information explicitly stated in the PDF.

GLOBAL RULES
- No placeholders ("Unknown", "N/A", "Not specified").
- If not explicitly stated -> null or [].
- Empty strings are NOT allowed.
- Do NOT infer access conditions from typical cohort practices, ethics statements, or general privacy language unless explicit.
- Do NOT copy values from other passes unless explicitly stated in the PDF.

--------------------------------------------------
informed_consent_type (STRICT dropdown; single select)
--------------------------------------------------
Allowed values (exact strings only):
- "Study specific consent"
- "Broad consent"
- "Passive/tacit consent"
- "No consent"

Rules:
- Set ONLY if the PDF explicitly states the type of consent for the data resource.
- If consent language is present but does not match one of the allowed values exactly -> null.
- Do NOT infer "Broad consent" from general biobank language unless explicitly stated.

--------------------------------------------------
access_rights (STRICT dropdown; single select)
--------------------------------------------------
Allowed values (exact strings only):
- "Open access"
- "Restricted access"
- "Non public"

Rules:
- "Open access" ONLY if explicitly stated data are publicly accessible without restrictions.
- "Restricted access" ONLY if explicitly stated access requires approval/application/DAC/contract.
- "Non public" ONLY if explicitly stated as not publicly accessible or closed.
- Statements like "data available on/upon request" or "available from corresponding author on reasonable request"
  count as "Restricted access".
- If not explicitly stated -> null.
NOTE: The UI may default to non-public, but extraction must not default; only extract explicit statements.
- IMPORTANT: publication status text (e.g., "Open access article", CC-BY license for the paper) is NOT evidence
  for resource data access_rights.

--------------------------------------------------
data_access_conditions (STRICT dropdown; multi-select)
--------------------------------------------------
Allowed values (exact strings only):
- "no restriction"
- "general research use"
- "health or medical or biomedical research"

Rules:
- Add an item ONLY if explicitly stated.
- If none are explicitly stated -> [].
- Do NOT infer from the topic area of the cohort.

--------------------------------------------------
data_use_conditions (STRICT dropdown; multi-select)
--------------------------------------------------
Allowed values (exact strings only):
- "research specific restrictions"
- "no general methods research"
- "genetic studies only"
- "not for profit, non commercial use only"
- "publication required"
- "collaboration required"
- "ethics approval required"
- "geographical restriction"
- "publication moratorium"
- "time limit on use"
- "user specific restriction"
- "project specific restriction"
- "institution specific restriction"
- "return to database or resource"
- "clinical care use"

Rules:
- Add an item ONLY if explicitly stated in the PDF as a condition/requirement/limitation.
- If none are explicitly stated -> [].
- Do NOT infer "ethics approval required" just because ethics approval is mentioned for running the study.
    It must be explicitly stated as a requirement for DATA ACCESS or DATA USE by external users.

--------------------------------------------------
data_access_conditions_description (free text)
--------------------------------------------------
- Fill ONLY if the PDF explicitly describes access terms/conditions in text (e.g., “available upon request”, “via data access committee”, “requires MTA/DTA”).
- Use short verbatim or near-verbatim wording (max ~400 characters).
- If not explicit -> null.

--------------------------------------------------
data_access_fee (boolean)
--------------------------------------------------
Allowed values:
- true
- false
- null

Rules:
- true ONLY if the PDF explicitly states a fee/cost/charge is required for access.
- false ONLY if explicitly states no fee / free of charge.
- If not explicit -> null.

--------------------------------------------------
release_type (STRICT dropdown; single select)
--------------------------------------------------
Allowed values (exact strings only):
- "Continuous"
- "Closed dataset"
- "Annually"
- "Periodically"
- "Other release type"

Rules:
- Set ONLY if the PDF explicitly states the release cycle of the resource or dataset updates.
- Do NOT infer release type from recruitment period or follow-up schedule.
- If the release cycle is explicitly stated but does not match allowed values:
    -> set "Other release type" and describe the exact wording in release_description.

--------------------------------------------------
release_description (free text)
--------------------------------------------------
- Fill ONLY if the PDF explicitly provides release/update cycle details OR if release_type is "Other release type".
- Use the exact wording from the PDF where possible.
- If not explicit -> null.

```

### Deterministic New Instructions

```text
Return ONLY valid JSON matching the template. No extra text.

TASK
- Extract resource-level access and consent information.
- Use only access-condition statements explicitly tied to the resource or its data.
- Do not infer access policy from generic ethics language or article publication status.

GLOBAL RULES
- Use null for missing scalar fields and [] for missing lists.
- For choice fields, use exact labels from the PDF; the current allowed values are injected automatically below.

FIELDS
- `informed_consent_type`: Return the explicit consent-type label only. Use the exact explicit label from the PDF. If not explicit -> null. The current allowed values are injected automatically below. Source: Resources.informed consent type
- `data_access_conditions`: Return exact explicit data-access-condition labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Resources.data access conditions
- `data_use_conditions`: Return exact explicit data-use-condition labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Resources.data use conditions
- `data_access_conditions_description`: Use short explicit wording that describes data access or data use conditions. Use the explicit value from the PDF only. If not explicit -> null. Source: Resources.data access conditions description
- `data_access_fee`: Return true or false only when a fee statement is explicit. Return true or false only when explicitly stated. If not explicit -> null. Source: Resources.data access fee
- `release_type`: Return the explicit release-type label only. Use the exact explicit label from the PDF. If not explicit -> null. The current allowed values are injected automatically below. Source: Resources.release type
- `release_description`: Use the explicit wording that describes the release cycle. Use the explicit value from the PDF only. If not explicit -> null. Source: Resources.release description
```

### Final Instructions

```text
SCHEMA-DRIVEN UPDATE
- This task was regenerated because the EMX2 schema changed for `task_access_conditions`.
- Removed fields: access_rights
Return ONLY valid JSON matching the template. No extra text.

TASK
- Extract resource-level access and consent information.
- Use only access-condition statements explicitly tied to the resource or its data.
- Do not infer access policy from generic ethics language or article publication status.

GLOBAL RULES
- Use null for missing scalar fields and [] for missing lists.
- For choice fields, use exact labels from the PDF; the current allowed values are injected automatically below.

FIELDS
- `informed_consent_type`: Return the explicit consent-type label only. Use the exact explicit label from the PDF. If not explicit -> null. The current allowed values are injected automatically below. Source: Resources.informed consent type
- `data_access_conditions`: Return exact explicit data-access-condition labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Resources.data access conditions
- `data_use_conditions`: Return exact explicit data-use-condition labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Resources.data use conditions
- `data_access_conditions_description`: Use short explicit wording that describes data access or data use conditions. Use the explicit value from the PDF only. If not explicit -> null. Source: Resources.data access conditions description
- `data_access_fee`: Return true or false only when a fee statement is explicit. Return true or false only when explicitly stated. If not explicit -> null. Source: Resources.data access fee
- `release_type`: Return the explicit release-type label only. Use the exact explicit label from the PDF. If not explicit -> null. The current allowed values are injected automatically below. Source: Resources.release type
- `release_description`: Use the explicit wording that describes the release cycle. Use the explicit value from the PDF only. If not explicit -> null. Source: Resources.release description
```

## task_subpopulations

- LLM rewritten: no
- Added fields: subpopulations[].recruitment_channel
- Removed fields: (none)
- Changed fields: (none)

### Old Instructions

```text
Extract explicitly defined subpopulations only (arms/subcohorts/strata/case-control/derivation-validation). Do NOT infer.
If none are explicitly defined -> {"subpopulations": []}.

Rules:
- Evidence threshold: only output a subpopulation when the paper explicitly frames a distinct subgroup
  (e.g., "subpopulation", "arm", "group", "stratum", "intervention arm", "control arm", "derivation/validation cohort").
- Do NOT create subpopulations from generic site lists, acknowledgements, or overall trial logistics.

- name (required):
  Use an explicit subgroup label from the paper whenever available
  (e.g., "cases", "controls", "derivation cohort", "validation cohort", "intervention arm", "usual care arm").
  If the subgroup is clearly defined but not formally named, you MAY create a short label using ONLY explicit wording from the paper.
  If you cannot produce a stable label from explicit wording -> do not output that row.

- pid: only if explicitly stated for the subpopulation; else null. Do NOT invent.
- description (required): 1–3 sentences describing the subpopulation using paper wording; no inference.
- keywords (required): 5–12 terms to improve findability; try NOT to reuse words from description; no generic filler.

- number_of_participants:
  fill when a subgroup N is explicitly stated OR explicitly approximated
  (e.g., "about", "around", "approximately", "circa", "~", "ca.").
  If approximated, convert to the nearest integer and keep it as an estimate-derived value.
  If no explicit or approximate N is given -> null.

- counts:
  return [] if no explicit subpopulation count breakdown is given.
  Otherwise return one row per explicitly reported count row for this subpopulation.
  Target shape follows the cohort workbook:
    - age_group (STRICT dropdown; single select):
      Allowed values (exact strings only):
      - "Prenatal"
      - "All ages"
      - "Infant (0-23 months)"
      - "Newborn (0-1 months)"
      - "Infants and toddlers (2-23 months)"
      - "Child (2-12 years)"
      - "Adolescent (13-17 years)"
      - "Adult (18+ years)"
      - "Young adult (18-24 years)"
      - "Adult (25-44 years)"
      - "Middle-aged (45-64 years)"
      - "Aged (65+ years)"
      - "Aged (65-79 years)"
      - "Aged (80+ years)"
      Rules:
      - Set ONLY if explicitly stated and exactly matches one of the allowed values.
      - Otherwise null (use null for overall counts without an explicit age-group label).
    - n_total: explicit total participant count for that row only.
    - n_female: explicit female count for that row only.
    - n_male: explicit male count for that row only.
  Important:
    - Do NOT calculate totals from percentages.
    - Do NOT infer female/male counts from total.
    - Do NOT invent an age group.
    - Partial rows are allowed: keep missing cells null when the paper gives only some of these values.
    - If the paper reports multiple rows (e.g., age bands), keep them all.

- inclusion_start/inclusion_end: explicit calendar years for this subpopulation only; ongoing -> inclusion_end null.
- age_groups (STRICT dropdown; multi-select):
  Allowed values (exact strings only):
  - "Prenatal"
  - "All ages"
  - "Infant (0-23 months)"
  - "Newborn (0-1 months)"
  - "Infants and toddlers (2-23 months)"
  - "Child (2-12 years)"
  - "Adolescent (13-17 years)"
  - "Adult (18+ years)"
  - "Young adult (18-24 years)"
  - "Adult (25-44 years)"
  - "Middle-aged (45-64 years)"
  - "Aged (65+ years)"
  - "Aged (65-79 years)"
  - "Aged (80+ years)"
  Rules:
  - Add values ONLY if explicitly listed for this subpopulation or explicitly mappable from stated age ranges.
  - If not explicit -> [].
- age_min/age_max: integers only if explicit; “≥X” -> age_min=X, age_max=null.
- main_medical_condition/comorbidity: list explicit disease groups/codes or named comorbidities only; no code inference.
- countries/regions: only if explicitly tied to this subpopulation; normalize "The Netherlands"->"Netherlands".
- inclusion_criteria / exclusion_criteria (STRICT dropdown values):
  These are checkbox categories, not free-text criteria.
  Only output items that are explicitly stated for this subpopulation
  (or explicitly stated as shared with study-level criteria).
  Allowed values (exact strings only) for BOTH fields:
  - "Age group inclusion criterion"
  - "Age of majority inclusion criterion"
  - "BMI range inclusion criterion"
  - "Clinically relevant exposure inclusion criterion"
  - "Clinically relevant lifestyle inclusion criterion"
  - "Country of residence inclusion criteria"
  - "Defined population inclusion criterion"
  - "Ethnicity inclusion criterion"
  - "Family status inclusion criterion"
  - "Gravidity inclusion criterion"
  - "Health status inclusion criterion"
  - "Hospital patient inclusion criterion"
  - "Sex inclusion criterion"
  - "Use of medication inclusion criterion"
  Rules:
  - Apply same mapping logic to BOTH inclusion_criteria and exclusion_criteria.
  - Although labels contain "inclusion", they are also used as exclusion categories.
  - If explicit criterion text exists but no allowed category matches, keep category out and put text in `other_*_criteria`.
  - If not explicit -> [].
- other_inclusion_criteria / other_exclusion_criteria:
  Keep explicit wording only (short phrases/sentences), no summaries, no inference.
- issued/modified: only if explicit publish/last modified dates for the subpopulation data; else null.
- theme (required): must include "Health". Allowed: "Health","Agriculture","Environment","Energy","Government and public sector".
- access_rights (required; STRICT dropdown; single select)
  Allowed values (exact strings only):
  - "Open access"
  - "Restricted access"
  - "Non public"
  Rules:
  - "Open access" ONLY if explicitly public.
  - "Restricted access" ONLY if explicitly controlled/restricted (including "available on request").
  - Otherwise default "Non public".
  - IMPORTANT: article license text (e.g., "Open access article", CC-BY) is NOT evidence for dataset access_rights.
- applicable_legislation (required; STRICT list)
  Allowed values (exact strings only):
  - "Data Governance Act"
  Rules:
  - Add "Data Governance Act" ONLY if explicitly stated in the PDF as applicable/mandating legislation.
  - Otherwise -> [].
  - Do NOT auto-add defaults.

```

### Deterministic New Instructions

```text
Return ONLY valid JSON matching the template. No extra text.

TASK
- Extract explicitly defined subpopulations only.
- Return one row per explicit arm, subgroup, cohort split or named subpopulation.
- Do not create subpopulations from incidental site lists, acknowledgements or generic participant descriptions.

GLOBAL RULES
- If no explicit subpopulations are defined, return {"subpopulations": []}.
- Only include a row when the subgroup is explicitly framed as distinct in the PDF.

FIELDS
- `subpopulations[]`: Return one object per explicit subpopulation. If none are explicit, return [].
  - `subpopulations[].name`: Use the explicit subgroup label from the PDF. Only include a row when this field is explicit. Source: Subpopulations.name
  - `subpopulations[].pid`: Only if an explicit persistent identifier is stated. Use the explicit value from the PDF only. If not explicit -> null. Source: Subpopulations.pid
  - `subpopulations[].description`: Use the explicit subgroup description. Use the explicit value from the PDF only. If not explicit -> null. Source: Subpopulations.description
  - `subpopulations[].keywords`: Return explicit retrieval keywords for this subgroup. Return a list of explicit values only. If not explicit -> []. Source: Subpopulations.keywords
  - `subpopulations[].number_of_participants`: Return the explicit participant count for this subgroup only. Return an integer only when explicitly stated. If not explicit -> null. Source: Subpopulations.number of participants
  - `subpopulations[].counts[]`: Output one row per explicit subgroup count row. If none are explicitly reported, return [].
    - `subpopulations[].counts[].age_group`: Only include an explicit age-group label for this reported count row. Use the exact explicit label from the PDF. If not explicit -> null. The current allowed values are injected automatically below. Source: Subpopulation counts.age group
    - `subpopulations[].counts[].n_total`: Explicit total participant count for this row only. Return an integer only when explicitly stated. If not explicit -> null. Source: Subpopulation counts.N total
    - `subpopulations[].counts[].n_female`: Explicit female count for this row only. Return an integer only when explicitly stated. If not explicit -> null. Source: Subpopulation counts.N female
    - `subpopulations[].counts[].n_male`: Explicit male count for this row only. Return an integer only when explicitly stated. If not explicit -> null. Source: Subpopulation counts.N male
  - `subpopulations[].inclusion_start`: Return an explicit calendar year only. Return an integer only when explicitly stated. If not explicit -> null. Source: Subpopulations.inclusion start
  - `subpopulations[].inclusion_end`: Return an explicit calendar year only. Return an integer only when explicitly stated. If not explicit -> null. Source: Subpopulations.inclusion end
  - `subpopulations[].age_groups`: Return exact age-group labels only when explicitly stated. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Subpopulations.age groups
  - `subpopulations[].age_min`: Return an explicit minimum age only. Return an integer only when explicitly stated. If not explicit -> null. Source: Subpopulations.age min
  - `subpopulations[].age_max`: Return an explicit maximum age only. Return an integer only when explicitly stated. If not explicit -> null. Source: Subpopulations.age max
  - `subpopulations[].main_medical_condition`: Return explicit disease or condition labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Subpopulations.main medical condition
  - `subpopulations[].comorbidity`: Return explicit comorbidity labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Subpopulations.comorbidity
  - `subpopulations[].countries`: Return countries explicitly tied to this subgroup. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Subpopulations.countries
  - `subpopulations[].regions`: Return regions explicitly tied to this subgroup. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Subpopulations.regions
  - `subpopulations[].inclusion_criteria`: Return exact explicit inclusion-criteria category labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Subpopulations.inclusion criteria
  - `subpopulations[].other_inclusion_criteria`: Copy explicit inclusion wording not captured by the category labels. Use the explicit value from the PDF only. If not explicit -> null. Source: Subpopulations.other inclusion criteria
  - `subpopulations[].exclusion_criteria`: Return exact explicit exclusion-criteria category labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Subpopulations.exclusion criteria
  - `subpopulations[].other_exclusion_criteria`: Copy explicit exclusion wording not captured by the category labels. Use the explicit value from the PDF only. If not explicit -> null. Source: Subpopulations.other exclusion criteria
  - `subpopulations[].issued`: Only if explicitly stated as subpopulation metadata. Return the explicit date or datetime string from the PDF. Do not infer. If not explicit -> null. Source: Subpopulations.issued
  - `subpopulations[].modified`: Only if explicitly stated as subpopulation metadata. Return the explicit date or datetime string from the PDF. Do not infer. If not explicit -> null. Source: Subpopulations.modified
  - `subpopulations[].theme`: Return explicit theme labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Subpopulations.theme
  - `subpopulations[].access_rights`: Return the explicit access-rights label only. Use the exact explicit label from the PDF. If not explicit -> null. The current allowed values are injected automatically below. Source: Subpopulations.access rights
  - `subpopulations[].applicable_legislation`: Return explicit legislation labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Subpopulations.applicable legislation
  - `subpopulations[].recruitment_channel`: Auto-added from the current schema. Use the explicit value from the PDF only. If not explicit -> null. Source: Subpopulations.recruitment channel
```

### Final Instructions

```text
SCHEMA-DRIVEN UPDATE
- This task was regenerated because the EMX2 schema changed for `task_subpopulations`.
- Added fields: subpopulations[].recruitment_channel
Return ONLY valid JSON matching the template. No extra text.

TASK
- Extract explicitly defined subpopulations only.
- Return one row per explicit arm, subgroup, cohort split or named subpopulation.
- Do not create subpopulations from incidental site lists, acknowledgements or generic participant descriptions.

GLOBAL RULES
- If no explicit subpopulations are defined, return {"subpopulations": []}.
- Only include a row when the subgroup is explicitly framed as distinct in the PDF.

FIELDS
- `subpopulations[]`: Return one object per explicit subpopulation. If none are explicit, return [].
  - `subpopulations[].name`: Use the explicit subgroup label from the PDF. Only include a row when this field is explicit. Source: Subpopulations.name
  - `subpopulations[].pid`: Only if an explicit persistent identifier is stated. Use the explicit value from the PDF only. If not explicit -> null. Source: Subpopulations.pid
  - `subpopulations[].description`: Use the explicit subgroup description. Use the explicit value from the PDF only. If not explicit -> null. Source: Subpopulations.description
  - `subpopulations[].keywords`: Return explicit retrieval keywords for this subgroup. Return a list of explicit values only. If not explicit -> []. Source: Subpopulations.keywords
  - `subpopulations[].number_of_participants`: Return the explicit participant count for this subgroup only. Return an integer only when explicitly stated. If not explicit -> null. Source: Subpopulations.number of participants
  - `subpopulations[].counts[]`: Output one row per explicit subgroup count row. If none are explicitly reported, return [].
    - `subpopulations[].counts[].age_group`: Only include an explicit age-group label for this reported count row. Use the exact explicit label from the PDF. If not explicit -> null. The current allowed values are injected automatically below. Source: Subpopulation counts.age group
    - `subpopulations[].counts[].n_total`: Explicit total participant count for this row only. Return an integer only when explicitly stated. If not explicit -> null. Source: Subpopulation counts.N total
    - `subpopulations[].counts[].n_female`: Explicit female count for this row only. Return an integer only when explicitly stated. If not explicit -> null. Source: Subpopulation counts.N female
    - `subpopulations[].counts[].n_male`: Explicit male count for this row only. Return an integer only when explicitly stated. If not explicit -> null. Source: Subpopulation counts.N male
  - `subpopulations[].inclusion_start`: Return an explicit calendar year only. Return an integer only when explicitly stated. If not explicit -> null. Source: Subpopulations.inclusion start
  - `subpopulations[].inclusion_end`: Return an explicit calendar year only. Return an integer only when explicitly stated. If not explicit -> null. Source: Subpopulations.inclusion end
  - `subpopulations[].age_groups`: Return exact age-group labels only when explicitly stated. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Subpopulations.age groups
  - `subpopulations[].age_min`: Return an explicit minimum age only. Return an integer only when explicitly stated. If not explicit -> null. Source: Subpopulations.age min
  - `subpopulations[].age_max`: Return an explicit maximum age only. Return an integer only when explicitly stated. If not explicit -> null. Source: Subpopulations.age max
  - `subpopulations[].main_medical_condition`: Return explicit disease or condition labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Subpopulations.main medical condition
  - `subpopulations[].comorbidity`: Return explicit comorbidity labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Subpopulations.comorbidity
  - `subpopulations[].countries`: Return countries explicitly tied to this subgroup. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Subpopulations.countries
  - `subpopulations[].regions`: Return regions explicitly tied to this subgroup. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Subpopulations.regions
  - `subpopulations[].inclusion_criteria`: Return exact explicit inclusion-criteria category labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Subpopulations.inclusion criteria
  - `subpopulations[].other_inclusion_criteria`: Copy explicit inclusion wording not captured by the category labels. Use the explicit value from the PDF only. If not explicit -> null. Source: Subpopulations.other inclusion criteria
  - `subpopulations[].exclusion_criteria`: Return exact explicit exclusion-criteria category labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Subpopulations.exclusion criteria
  - `subpopulations[].other_exclusion_criteria`: Copy explicit exclusion wording not captured by the category labels. Use the explicit value from the PDF only. If not explicit -> null. Source: Subpopulations.other exclusion criteria
  - `subpopulations[].issued`: Only if explicitly stated as subpopulation metadata. Return the explicit date or datetime string from the PDF. Do not infer. If not explicit -> null. Source: Subpopulations.issued
  - `subpopulations[].modified`: Only if explicitly stated as subpopulation metadata. Return the explicit date or datetime string from the PDF. Do not infer. If not explicit -> null. Source: Subpopulations.modified
  - `subpopulations[].theme`: Return explicit theme labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Subpopulations.theme
  - `subpopulations[].access_rights`: Return the explicit access-rights label only. Use the exact explicit label from the PDF. If not explicit -> null. The current allowed values are injected automatically below. Source: Subpopulations.access rights
  - `subpopulations[].applicable_legislation`: Return explicit legislation labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Subpopulations.applicable legislation
  - `subpopulations[].recruitment_channel`: Auto-added from the current schema. Use the explicit value from the PDF only. If not explicit -> null. Source: Subpopulations.recruitment channel
```
