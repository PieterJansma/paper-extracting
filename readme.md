# paper-extracting

This repository is now cohort-only.

There is one supported production entrypoint:

```bash
bash src/run_cluster_cohort.sh -p all --pdfs data/oncolifes.pdf -o oncolifes_cohort.xlsx
```

## What This Pipeline Does

`src/run_cluster_cohort.sh` is the cluster wrapper around the cohort extraction pipeline. It does the following:

1. Creates a run directory under `logs/runs/<run_id>/`.
2. Copies `config.cohort.toml` to a run-specific `config.runtime.toml`.
3. Builds the run-specific prompt file `prompts.runtime.toml`.
4. Fetches the latest EMX2 ontology and model CSV files from `MOLGENIS_EMX2_REPO`.
5. Builds a live dynamic EMX2 runtime registry for the `UMCGCohortsStaging` profile.
6. Optionally runs prompt schema sync against the current EMX2 schema.
7. Starts the local LLM servers and load balancer.
8. Optionally prefetches OCR text for weak PDFs, or for all PDFs when forced.
9. Runs `src/main_cohort.py` to extract the PDF into a cohort workbook.
10. Normalizes extracted ontology values against the live EMX2 ontologies.
11. Writes run logs, issues, prompt artifacts, and optional OCR comparison files.

## Main Files

- `config.cohort.toml`: base runtime configuration for the cohort flow.
- `prompts/prompts_cohort.toml`: baseline prompt set.
- `src/run_cluster_cohort.sh`: supported cluster runner.
- `src/main_cohort.py`: main extractor and Excel writer.
- `src/cohort_prompt_schema_updater.py`: prompt schema sync and before/after diff generation.
- `src/cohort_dynamic_prompts.py`: fully dynamic prompt generation from the live EMX2 schema.
- `src/emx2_dynamic_runtime.py`: live EMX2 schema and ontology runtime builder.
- `src/fix_molgenis_staging_types_dynamic.py`: post-extraction datatype and ontology normalization.

## Default Production Run

Use this for the normal supported workflow:

```bash
bash src/run_cluster_cohort.sh -p all --pdfs data/oncolifes.pdf -o oncolifes_cohort.xlsx
```

What this does by default:

- `COHORT_DYNAMIC_EMX2_RUNTIME=1`
- `COHORT_PROMPT_SCHEMA_SYNC=1`
- `COHORT_PROMPT_SCHEMA_SYNC_LLM=1`
- `COHORT_DYNAMIC_PROMPTS=0`

That means:

- the run uses the live EMX2 schema and live ontology CSV files,
- the baseline prompt file is schema-synced when the schema changed,
- only changed prompt tasks are rewritten,
- the runtime uses `logs/runs/<run_id>/prompts.runtime.toml`,
- it does not replace the full prompt set with a fully auto-generated prompt set.

## Common Commands

Run all passes for one PDF:

```bash
bash src/run_cluster_cohort.sh -p all --pdfs data/oncolifes.pdf -o oncolifes_cohort.xlsx
```

Run only specific passes:

```bash
bash src/run_cluster_cohort.sh -p A C E --pdfs data/oncolifes.pdf -o oncolifes_partial.xlsx
```

Run on multiple PDFs:

```bash
bash src/run_cluster_cohort.sh -p all \
  --pdfs data/oncolifes.pdf data/another_paper.pdf \
  --paper-names oncolifes another_paper \
  -o multi_paper_cohort.xlsx
```

Force OCR prefetch for all selected PDFs:

```bash
bash src/run_cluster_cohort.sh --ocr -p all --pdfs data/oncolifes.pdf -o oncolifes_ocr_prefetch.xlsx
```

Force OCR prefetch and force OCR text to be used:

```bash
bash src/run_cluster_cohort.sh --ocr-force-use -p all --pdfs data/oncolifes.pdf -o oncolifes_ocr_forced.xlsx
```

Write pypdf/OCR text comparison files:

```bash
bash src/run_cluster_cohort.sh --ocr-dump -p all --pdfs data/oncolifes.pdf -o oncolifes_ocr_dump.xlsx
```

Use a custom EMX2 fork and branch:

```bash
MOLGENIS_EMX2_REPO="PieterJansma/molgenis-emx2" \
MOLGENIS_EMX2_REF="main" \
bash src/run_cluster_cohort.sh -p all --pdfs data/oncolifes.pdf -o oncolifes_fork.xlsx
```

Run the fully dynamic prompt route for testing:

```bash
MOLGENIS_EMX2_REPO="PieterJansma/molgenis-emx2" \
MOLGENIS_EMX2_REF="main" \
COHORT_PROMPT_SCHEMA_SYNC=0 \
COHORT_DYNAMIC_PROMPTS=1 \
bash src/run_cluster_cohort.sh -p all --pdfs data/oncolifes.pdf -o oncolifes_autogen_only.xlsx
```

Use the last command only when you explicitly want to inspect the fully auto-generated prompt set. It is useful for prompt-generation tests, but it is not the default production path.

## Supported CLI Arguments

`src/run_cluster_cohort.sh` forwards these arguments to `src/main_cohort.py`:

- `-p`, `--passes`: one or more passes. Valid values are `A`, `B`, `C`, `D`, `D1`, `D2`, `E`, `F`, `F1`, `F2`, `G`, `H`, `X2`, `X3`, `Y`, or `all`.
- `--pdfs`: one or more PDF paths.
- `--paper-names`: optional labels, same length as `--pdfs`.
- `-o`, `--output`: output workbook name.

Runner-only flags:

- `--ocr`: force OCR prefetch for all selected PDFs.
- `--ocr-force-use`: force OCR prefetch and prefer OCR text over pypdf text.
- `--ocr-dump`: write pypdf/OCR/diff/summary text files per PDF.
- `--ocr-dump-dir <dir>`: write OCR compare files to a custom directory.

## Output Files

The main output is the Excel workbook you pass with `-o`:

- `oncolifes_cohort.xlsx`

Additional output files may be written next to the workbook:

- `<output>.dynamic_prompts.toml`
  Written when `COHORT_DYNAMIC_PROMPTS=1`.
- `<output>.dynamic_emx2_registry.json`
  Written when dynamic runtime is enabled.
- `<output>.dynamic_prompt_constraints.json`
  Written when dynamic runtime is enabled.
- `<output>.issues.json`
  Fallback issues file if `PIPELINE_ISSUES_FILE` is not overridden.

The run directory always contains:

- `logs/runs/<run_id>/config.runtime.toml`
- `logs/runs/<run_id>/status.jsonl`
- `logs/runs/<run_id>/pipeline_issues.json`

The run directory also contains:

- `logs/runs/<run_id>/prompts.runtime.toml`
  The effective runtime prompt file used by the normal production route.

When prompt schema sync is enabled, the run directory additionally contains:

- `logs/runs/<run_id>/prompt_schema_sync.report.json`
- `logs/runs/<run_id>/prompt_schema_sync.compare.json`
- `logs/runs/<run_id>/prompt_schema_sync.compare.md`
- `logs/runs/<run_id>/prompt_schema_sync.prompt.diff`
- `logs/runs/<run_id>/prompt_schema_sync.before_after.md`
- `logs/runs/<run_id>/prompt_schema_sync.llm_report.json`
- `logs/runs/<run_id>/prompt_schema_sync.llm.compare.json`
- `logs/runs/<run_id>/prompt_schema_sync.llm.compare.md`
- `logs/runs/<run_id>/prompt_schema_sync.llm.prompt.diff`

When OCR compare dumping is enabled, the run directory additionally contains:

- `logs/runs/<run_id>/text_compare/`
- `logs/runs/<run_id>/ocr_prefetch/`
- `logs/runs/<run_id>/logs/ocr_prefetch_failed.txt` when OCR prefetch failed for any file

## Prompt Modes

There are two important prompt-generation modes.

### 1. Default Production Mode

This is the normal route:

- `COHORT_PROMPT_SCHEMA_SYNC=1`
- `COHORT_DYNAMIC_PROMPTS=0`

Behavior:

- start from `prompts/prompts_cohort.toml`,
- compare the current EMX2 schema with the cached schema state,
- rewrite only changed tasks,
- write the result to `logs/runs/<run_id>/prompts.runtime.toml`.

This is the file you usually inspect after a production run.

### 2. Fully Dynamic Prompt Mode

This is the test route:

- `COHORT_PROMPT_SCHEMA_SYNC=0`
- `COHORT_DYNAMIC_PROMPTS=1`

Behavior:

- build the prompt set directly from the current EMX2 runtime schema,
- replace task sections at runtime,
- write the generated prompt file to `<output>.dynamic_prompts.toml`.

This is the file you inspect when testing auto-generated tasks for new profile tables.

## Prompt Sync Artifacts

For prompt schema sync, the most useful human-readable artifact is usually:

```bash
LATEST_RUN="$(ls -td logs/runs/* | head -n 1)"
cat "$LATEST_RUN/prompt_schema_sync.before_after.md"
```

This file is intended to show the compact before/after prompt changes for the tasks that were actually rewritten.

## Prompt History

Persistent prompt history is stored under `prompts/history/`:

```text
prompts/history/
  baseline/
    prompts_cohort.toml
  2026/
    2026-04-07/
      20260407T153012_run7067294.prompt_change.md
      20260407T181500_run7067451.prompt_change.md
```

Rules:

- `prompts/prompts_cohort.toml` is the baseline source-of-truth prompt file.
- `prompts/history/baseline/prompts_cohort.toml` is the stored baseline snapshot.
- each later prompt change is archived as a compact markdown change file,
- multiple prompt changes on the same date are stored in the same date directory,
- date directories are grouped by year.

## Ontology Normalization

The cohort flow uses dynamic EMX2 post-processing after extraction.

Current behavior:

- `countries` and `regions` use deterministic matching first, then LLM selection from a short candidate list.
- other `ontology` and `ontology_array` fields use the same LLM fallback only when the ontology has more than `16` allowed values.

Relevant environment variables:

- `DYNAMIC_ONTOLOGY_LLM_FALLBACK=1`
- `DYNAMIC_ONTOLOGY_LLM_FALLBACK_THRESHOLD=16`
- `DYNAMIC_ONTOLOGY_LLM_MAX_CANDIDATES=5`
- `DYNAMIC_ONTOLOGY_LLM_MAX_LOOKUPS=50`

## Useful Environment Overrides

These usually do not need to be set manually, but they are available:

- `PDF_EXTRACT_CONFIG`
- `PDF_EXTRACT_PROMPTS`
- `MOLGENIS_EMX2_REPO`
- `MOLGENIS_EMX2_REF`
- `COHORT_PROMPT_SCHEMA_SYNC`
- `COHORT_PROMPT_SCHEMA_SYNC_LLM`
- `COHORT_DYNAMIC_EMX2_RUNTIME`
- `COHORT_DYNAMIC_PROMPTS`
- `COHORT_PROMPT_SCHEMA_BASE_CSV`
- `COHORT_PROMPT_SCHEMA_STATE_DIR`
- `COHORT_PROMPT_SCHEMA_HISTORY_DIR`
- `AUTO_FETCH_EMX2_ONTOLOGIES`
- `OCR_VLM_ENABLE`
- `STRIP_REFERENCES`

## Quick Checks After a Run

Show the latest run directory:

```bash
LATEST_RUN="$(ls -td logs/runs/* | head -n 1)"
echo "$LATEST_RUN"
```

List run artifacts:

```bash
LATEST_RUN="$(ls -td logs/runs/* | head -n 1)"
ls "$LATEST_RUN"
```

Read the latest prompt before/after summary:

```bash
LATEST_RUN="$(ls -td logs/runs/* | head -n 1)"
cat "$LATEST_RUN/prompt_schema_sync.before_after.md"
```

Fetch the latest runtime prompt file from the cluster to your Mac:

```bash
rsync -avhP \
  tunnel+nibbler:/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/paper-extracting/logs/runs/$(ssh tunnel+nibbler 'cd /groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/paper-extracting && ls -td logs/runs/* | head -n 1 | xargs basename')/prompts.runtime.toml \
  "/Users/p.jansma/Documents/cluster_data/"
```

Fetch the latest fully dynamic prompt file from the cluster to your Mac:

```bash
rsync -avhP \
  tunnel+nibbler:/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/paper-extracting/oncolifes_autogen_only.xlsx.dynamic_prompts.toml \
  "/Users/p.jansma/Documents/cluster_data/"
```
