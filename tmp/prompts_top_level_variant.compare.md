# Prompt Schema Update Comparison

## task_access_conditions

- LLM rewritten: no
- Added fields: data_access_url
- Removed fields: (none)
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
- `access_rights`: Return the explicit access-rights label only. Use the exact explicit label from the PDF. If not explicit -> null. The current allowed values are injected automatically below. Source: Resources.access rights
- `data_access_conditions`: Return exact explicit data-access-condition labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Resources.data access conditions
- `data_use_conditions`: Return exact explicit data-use-condition labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Resources.data use conditions
- `data_access_conditions_description`: Use short explicit wording that describes data access or data use conditions. Use the explicit value from the PDF only. If not explicit -> null. Source: Resources.data access conditions description
- `data_access_fee`: Return true or false only when a fee statement is explicit. Return true or false only when explicitly stated. If not explicit -> null. Source: Resources.data access fee
- `release_type`: Return the explicit release-type label only. Use the exact explicit label from the PDF. If not explicit -> null. The current allowed values are injected automatically below. Source: Resources.release type
- `release_description`: Use the explicit wording that describes the release cycle. Use the explicit value from the PDF only. If not explicit -> null. Source: Resources.release description
- `data_access_url`: Auto-added from the current schema. Use the explicit value from the PDF only. If not explicit -> null. Source: Resources.data access url
```

### Final Instructions

```text
SCHEMA-DRIVEN UPDATE
- This task was regenerated because the EMX2 schema changed for `task_access_conditions`.
- Added fields: data_access_url
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
- `access_rights`: Return the explicit access-rights label only. Use the exact explicit label from the PDF. If not explicit -> null. The current allowed values are injected automatically below. Source: Resources.access rights
- `data_access_conditions`: Return exact explicit data-access-condition labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Resources.data access conditions
- `data_use_conditions`: Return exact explicit data-use-condition labels only. Return exact explicit labels only. If not explicit -> []. The current allowed values are injected automatically below. Source: Resources.data use conditions
- `data_access_conditions_description`: Use short explicit wording that describes data access or data use conditions. Use the explicit value from the PDF only. If not explicit -> null. Source: Resources.data access conditions description
- `data_access_fee`: Return true or false only when a fee statement is explicit. Return true or false only when explicitly stated. If not explicit -> null. Source: Resources.data access fee
- `release_type`: Return the explicit release-type label only. Use the exact explicit label from the PDF. If not explicit -> null. The current allowed values are injected automatically below. Source: Resources.release type
- `release_description`: Use the explicit wording that describes the release cycle. Use the explicit value from the PDF only. If not explicit -> null. Source: Resources.release description
- `data_access_url`: Auto-added from the current schema. Use the explicit value from the PDF only. If not explicit -> null. Source: Resources.data access url
```
