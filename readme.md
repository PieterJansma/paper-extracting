# Paper Extracting

Deze repository extraheert gestructureerde metadata uit papers naar Excel-workbooks. Er zijn drie duidelijke routes:

- `final`: algemene pipeline met de baseline promptset
- `cohort`: cohort-specifieke pipeline met een eigen baseline promptset
- `dynamic cohort`: cohort-route die live schema- en ontology-wijzigingen doorrekent vanaf de cohort-basisprompt

Alle commands hieronder gaan ervan uit dat je vanuit de repo-root runt.

## Structuur

- `config.final.toml`
  runtime-instellingen voor LLM, PDF en timeouts
- `prompts/prompts.toml`
  baseline promptset voor de algemene/final pipeline
- `prompts/prompts_cohort.toml`
  cohort-specifieke baseline promptset
- `prompts/archive/`
  oudere of experimentele promptvarianten om mee te vergelijken
- `schemas/molgenis_UMCGCohortsStaging.csv`
  schema-export die de dynamische cohort-route als bron van waarheid gebruikt
- `schemas/molgenis_UMCGCohortsStaging.xlsx`
  workbook voor post-write datatype-normalisatie
- `src/main_final.py`
  extractieflow voor de algemene/final pipeline
- `src/main_cohort.py`
  extractieflow voor de cohort-pipeline
- `src/run_cluster_final.sh`
  cluster-runner voor de final pipeline
- `src/run_cluster_cohort.sh`
  cluster-runner voor de cohort-pipeline
- `src/run_prompt_schema_demo.sh`
  demo-runner die laat zien hoe schemawijzigingen prompt-updates veroorzaken
- `tests/prompt_schema_demo/`
  kleine nep-EMX2 fixture voor schema-diff en Qwen prompt rewrite tests

## Promptmodel

De promptlogica is nu expliciet in lagen verdeeld:

1. `prompts/prompts.toml` is de baseline voor de algemene/final route.
2. `prompts/prompts_cohort.toml` is de baseline voor de cohort-route.
3. De dynamische cohort-route start altijd vanuit `prompts/prompts_cohort.toml`.
4. Daarna vergelijkt hij het actuele `UMCGCohortsStaging` schema met de bekende baseline.
5. Alleen de geraakte tasks worden aangepast; ongewijzigde tasks blijven exact uit de bestaande cohort-prompt komen.

Kort:

- menselijke baseline blijft leidend
- schema bepaalt wat moet veranderen
- ontology/ref CSV’s leveren de actuele choices en referenties
- Qwen herschrijft alleen de changed tasks of changed field blocks

## Route 1: Final Pipeline

Gebruik deze route voor de algemene, niet-cohort-specifieke extractieflow.

Lokale run:

```bash
pdf-extract -p all -o final_result.xlsx
```

Cluster-run:

```bash
bash src/run_cluster_final.sh -p all --pdfs data/*.pdf -o final_all.xlsx
```

Belangrijk:

- promptbron: `prompts/prompts.toml`
- hoofdscript: `src/main_final.py`
- cluster-runner: `src/run_cluster_final.sh`

## Route 2: Cohort Pipeline

Dit is de normale cohortflow zonder automatische schema-sync.

Lokale/cluster-run:

```bash
bash src/run_cluster_cohort.sh -p all --pdfs data/oncolifes.pdf -o oncolifes_cohort.xlsx
```

Belangrijk:

- promptbron: `prompts/prompts_cohort.toml`
- hoofdscript: `src/main_cohort.py`
- cluster-runner: `src/run_cluster_cohort.sh`
- default gedrag: gebruikt gewoon de bestaande cohort-basisprompt

## Route 3: Dynamische Cohort Promptflow

Deze route is bedoeld voor schema-gedreven promptupdates.

Hij doet dit:

1. leest `schemas/molgenis_UMCGCohortsStaging.csv`
2. haalt de relevante live EMX2 ontology/ref CSV’s op
3. vergelijkt live schema tegen de baseline
4. houdt `prompts/prompts_cohort.toml` als startpunt
5. past alleen changed tasks aan
6. laat Qwen die changed tasks of field blocks herschrijven

Run op cluster:

```bash
COHORT_DYNAMIC_EMX2_RUNTIME=1 \
COHORT_DYNAMIC_PROMPTS=0 \
COHORT_PROMPT_SCHEMA_SYNC=1 \
COHORT_PROMPT_SCHEMA_SYNC_LLM=1 \
bash src/run_cluster_cohort.sh -p all --pdfs data/oncolifes.pdf -o oncolifes_dynamic.xlsx
```

Zonder Qwen-polish:

```bash
COHORT_DYNAMIC_EMX2_RUNTIME=1 \
COHORT_DYNAMIC_PROMPTS=0 \
COHORT_PROMPT_SCHEMA_SYNC=1 \
COHORT_PROMPT_SCHEMA_SYNC_LLM=0 \
bash src/run_cluster_cohort.sh -p all --pdfs data/oncolifes.pdf -o oncolifes_dynamic.xlsx
```

Belangrijk:

- basistemplate blijft `prompts/prompts_cohort.toml`
- de uiteindelijke runtime-prompt wordt per run berekend
- changed tasks zijn terug te zien in `logs/runs/<run_id>/`

## Dynamic Fetch Flow (English)

This is the exact idea behind the dynamic cohort route.

The starting point is always:

- baseline prompt: `prompts/prompts_cohort.toml`
- baseline schema: `schemas/molgenis_UMCGCohortsStaging.csv`
- active profile: `UMCGCohortsStaging`

### Step 1: A Python helper decides what must be fetched

The first decision is not made in shell; it is made by `src/emx2_dynamic_runtime.py`.

That script:

1. reads the shared EMX2 model rows for profile `UMCGCohortsStaging`
2. filters them to the cohort runtime tables
3. inspects every field
4. derives which extra CSV files are required:
   - ontology CSVs such as `data/_ontologies/Access rights.csv`
   - model/ref CSVs such as `data/_models/shared/Organisations.csv`
5. exports a combined live schema CSV when needed

The key commands are:

```bash
python3 src/emx2_dynamic_runtime.py required-paths --mode cohort
python3 src/emx2_dynamic_runtime.py model-paths --mode cohort
python3 src/emx2_dynamic_runtime.py export-schema-csv --profile UMCGCohortsStaging --output tmp/live_schema.csv
```

So the idea is:

- Python decides what is needed
- shell only fetches what Python says is needed

### Step 2: `run_cluster_cohort.sh` fetches those files

`src/run_cluster_cohort.sh` then calls that helper and fetches the files from `molgenis/molgenis-emx2`.

It does this in two layers:

1. a small fixed bootstrap set:
   - `Countries.csv`
   - `Regions.csv`
   - `Resources.csv`
   - `Organisations.csv`
   - `Subpopulations.csv`
2. the dynamic set returned by:
   - `required-paths --mode cohort`
   - `model-paths --mode cohort`

The actual fetch function is `fetch_emx2_csv()` inside `src/run_cluster_cohort.sh`.
It tries the configured Git ref first, then falls back to `main`, then `master`, and stores the fetched files under the per-run cache:

- `logs/runs/<run_id>/emx2_cache/repo/...`

### Step 3: The live schema is exported and checked

After fetching, `run_cluster_cohort.sh` exports a fresh live schema with:

```bash
python3 src/emx2_dynamic_runtime.py export-schema-csv \
  --profile UMCGCohortsStaging \
  --local-root "$MOLGENIS_EMX2_LOCAL_ROOT" \
  --output "$LIVE_SCHEMA_CSV"
```

That exported file is the current live view of the profile, built from the fetched shared model CSVs.

Then the script checks:

1. is the live schema identical to the cached previous live schema?
2. if yes:
   - reuse the cached synced prompt
3. if no:
   - rebuild the synced prompt from the cohort baseline

### Step 4: The prompt is rebuilt only where needed

If the live schema changed, `src/cohort_prompt_schema_updater.py` is used.

It compares:

- old schema: `schemas/molgenis_UMCGCohortsStaging.csv`
- new live schema: exported live schema from the fetched EMX2 files
- base prompt: `prompts/prompts_cohort.toml`

The updater then:

1. keeps unchanged tasks exactly as they were
2. updates only changed tasks
3. removes fields that disappeared from the schema
4. adds fields that were added to the schema
5. updates ontology choices when the source ontology CSV changed

If `COHORT_PROMPT_SCHEMA_SYNC_LLM=1`, Qwen is then used only on those changed tasks or changed field blocks to improve readability while preserving the old prompt style.

### Step 5: `main_cohort.py` uses that runtime prompt

After the schema-sync step, `src/main_cohort.py` runs with:

- the runtime config
- the runtime prompt chosen by the previous steps

So the extraction model never guesses which live files matter.
That decision was already made by `emx2_dynamic_runtime.py`.

## Expected Example

Here is the expected idea of a real change:

- `Resources.csv`
  - removes `release description`
  - adds `data access url`
- `Access rights.csv`
  - adds `Embargoed access`
- `Release types.csv`
  - adds `Quarterly`

Expected result:

- `task_access_conditions` changes
  - `release_description` disappears
  - `data_access_url` is added
  - `access_rights` gets the new allowed value `Embargoed access`
  - `release_type` gets the new allowed value `Quarterly`
- unrelated tasks such as `task_overview` stay unchanged

That is exactly the behavior demonstrated by:

```bash
PROMPT_SCHEMA_DEMO_WITH_LLM=1 bash src/run_prompt_schema_demo.sh
```

## Useful Inspection Commands

If you want to show the flow clearly to someone else, these are the most useful commands:

Show what Python thinks must be fetched:

```bash
python3 src/emx2_dynamic_runtime.py required-paths --mode cohort
python3 src/emx2_dynamic_runtime.py model-paths --mode cohort
```

Show the exported live schema:

```bash
python3 src/emx2_dynamic_runtime.py export-schema-csv \
  --profile UMCGCohortsStaging \
  --output tmp/live_schema.csv
```

Run the real dynamic cohort route:

```bash
COHORT_DYNAMIC_EMX2_RUNTIME=1 \
COHORT_DYNAMIC_PROMPTS=0 \
COHORT_PROMPT_SCHEMA_SYNC=1 \
COHORT_PROMPT_SCHEMA_SYNC_LLM=1 \
bash src/run_cluster_cohort.sh -p all --pdfs data/oncolifes.pdf -o oncolifes_dynamic.xlsx
```

Show what changed in the prompt:

```bash
LATEST_RUN="$(ls -td logs/runs/* | head -n 1)"
cat "$LATEST_RUN/prompt_schema_sync.compare.md"
cat "$LATEST_RUN/prompt_schema_sync.llm.compare.md"
```

## Prompt Demo Route

Gebruik deze om aan anderen te laten zien wat een schemawijziging doet met de prompt.

Deterministische demo:

```bash
bash src/run_prompt_schema_demo.sh
```

Demo met Qwen rewrite:

```bash
PROMPT_SCHEMA_DEMO_WITH_LLM=1 bash src/run_prompt_schema_demo.sh
```

Belangrijkste demo-outputs:

- `tmp/prompt_schema_demo/base_schema.csv`
- `tmp/prompt_schema_demo/variant_schema.csv`
- `tmp/prompt_schema_demo/base_dynamic.toml`
- `tmp/prompt_schema_demo/updated_from_existing.toml`
- `tmp/prompt_schema_demo/prompt_update.compare.md`
- `tmp/prompt_schema_demo/prompt_update.summary.md`

De demo gebruikt bewust:

- bestaande basisprompt: `prompts/prompts_cohort.toml`
- kleine fixture: `tests/prompt_schema_demo/`

## Outputs

Cluster-runs schrijven hun artifacts onder:

- `logs/runs/<run_id>/`

Daar vind je onder andere:

- runtime config
- runtime prompt
- status logs
- compare-bestanden voor schema-sync
- pipeline issues

De prompt-schema demo schrijft tijdelijke review-output onder:

- `tmp/prompt_schema_demo/`

## Meest Aangepaste Bestanden

- `prompts/prompts.toml`
  baseline promptset voor final
- `prompts/prompts_cohort.toml`
  baseline promptset voor cohort
- `schemas/molgenis_UMCGCohortsStaging.csv`
  bron van waarheid voor dynamische cohort prompt-sync
- `src/main_final.py`
  final extractieflow
- `src/main_cohort.py`
  cohort extractieflow
- `src/cohort_prompt_schema_updater.py`
  schema-diff prompt updater
- `src/run_cluster_final.sh`
  final cluster-runner
- `src/run_cluster_cohort.sh`
  cohort cluster-runner

## Troubleshooting

- prompts worden niet gevonden
  zet `PDF_EXTRACT_PROMPTS` of gebruik de standaardpaden onder `prompts/`
- schema-sync lijkt niets te doen
  check of `changed_tasks=0`; dan was het live schema niet veranderd
- cohort normalisatie vindt het schema-workbook niet
  zet `--schema-xlsx` of `MOLGENIS_SCHEMA_XLSX`, anders gebruikt de code `schemas/molgenis_UMCGCohortsStaging.xlsx`
- demo-output is onduidelijk
  open `tmp/prompt_schema_demo/prompt_update.summary.md`; dat bestand is bedoeld voor menselijke review
