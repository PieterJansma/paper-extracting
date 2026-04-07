# paper-extracting

Dit repository is nu cohort-only.

Er is nog maar één ondersteunde productieroute:

```bash
bash src/run_cluster_cohort.sh -p all --pdfs data/oncolifes.pdf -o oncolifes_cohort.xlsx
```

## Belangrijkste bestanden

- `config.cohort.toml`: basisconfig voor de cohort-run
- `prompts/prompts_cohort.toml`: baseline promptset
- `src/run_cluster_cohort.sh`: enige ondersteunde cluster-entrypoint
- `src/main_cohort.py`: cohort extractie naar Excel
- `src/cohort_prompt_schema_updater.py`: prompt schema sync
- `src/cohort_runtime_utils.py`: gedeelde extractiehelpers voor de cohortflow

## Runtime flow

`src/run_cluster_cohort.sh` voert in productie deze stappen uit:

1. Live EMX2 cohort-CSV's ophalen uit `MOLGENIS_EMX2_REPO`.
2. Een actuele schema-export maken voor `UMCGCohortsStaging`.
3. De nieuwe schema-export vergelijken met de vorige cached schema-state.
4. Alleen de geraakte promptvelden bijwerken.
5. Een leesbare `before/after` promptsamenvatting schrijven.
6. `main_cohort.py` uitvoeren.
7. Het workbook normaliseren tegen de live ontologies.

## Prompt-sync artifacts

Per run komen de prompt-artifacts in `logs/runs/<run_id>/`:

- `prompt_schema_sync.compare.md`
- `prompt_schema_sync.prompt.diff`
- `prompt_schema_sync.llm.compare.md`
- `prompt_schema_sync.llm.prompt.diff`
- `prompt_schema_sync.before_after.md`

De meest bruikbare presentatie-output is meestal:

```bash
LATEST_RUN="$(ls -td logs/runs/* | head -n 1)"
cat "$LATEST_RUN/prompt_schema_sync.before_after.md"
```

## Prompt history

De blijvende promptgeschiedenis staat onder `prompts/history/`:

```text
prompts/history/
  baseline/
    prompts_cohort.toml
  2026/
    2026-04-07/
      20260407T153012_run7067294.prompt_change.md
      20260407T181500_run7067451.prompt_change.md
```

Regels:

- de baseline wordt eenmalig vastgelegd vanuit de huidige `prompts/prompts_cohort.toml`
- daarna wordt per wijziging alleen de compacte `before/after` markdown opgeslagen
- meerdere wijzigingen op dezelfde dag komen in dezelfde datummap
- datummappen zitten per jaar gegroepeerd

## Ontology normalisatie

De cohortflow doet nu twee lagen normalisatie:

- `countries` en `regions`: eerst deterministic matching, daarna LLM-keuze uit een korte kandidatenlijst.
- overige `ontology` en `ontology_array` velden: dezelfde fallback, maar alleen wanneer de ontology groter is dan `30` keuzes.

Relevante env-vars:

- `DYNAMIC_ONTOLOGY_LLM_FALLBACK=1`
- `DYNAMIC_ONTOLOGY_LLM_FALLBACK_THRESHOLD=30`
- `DYNAMIC_ONTOLOGY_LLM_MAX_CANDIDATES=5`
- `DYNAMIC_ONTOLOGY_LLM_MAX_LOOKUPS=50`

## Nuttige overrides

Je hoeft deze normaal niet te zetten, want de cohort-run defaults staan al goed. Ze blijven wel beschikbaar:

- `PDF_EXTRACT_CONFIG`
- `PDF_EXTRACT_PROMPTS`
- `MOLGENIS_EMX2_REPO`
- `MOLGENIS_EMX2_REF`
- `COHORT_PROMPT_SCHEMA_SYNC`
- `COHORT_PROMPT_SCHEMA_SYNC_LLM`
- `COHORT_DYNAMIC_EMX2_RUNTIME`
- `COHORT_DYNAMIC_PROMPTS`

## Snelle controle na een run

```bash
LATEST_RUN="$(ls -td logs/runs/* | head -n 1)"
ls "$LATEST_RUN"
cat "$LATEST_RUN/prompt_schema_sync.before_after.md"
```
