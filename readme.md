# Paper Extracting (Final Pipeline)

This repo extracts structured fields from PDFs using a local OpenAI-compatible LLM server.
The *final* workflow is centered around these three files:

- `config.final.toml` (all extraction rules + templates)
- `src/main_final.py` (runs the multi-pass extractor and writes Excel)
- `src/run_cluster_final.sh` (cluster launcher + load balancer + extraction)

---

## 1) `config.final.toml` (the source of truth)

`config.final.toml` defines **all prompts, schemas, and rules** for the final pipeline. It is divided into passes, each one extracting a specific section.

### LLM + PDF settings

```
[llm]
base_url = "http://127.0.0.1:8080/v1"
model    = "numind/NuExtract-2.0-8B"
api_key  = "sk-local"
use_grammar = false
max_tokens = 6000
temperature = 0.0

[pdf]
path = "data/concrete.pdf"
max_pages = 40
```

### Passes (what gets extracted)

The final config uses these passes (names must match in `main_final.py`):

- **A: Overview** -> `[task_overview]`
- **B: Design & structure** -> `[task_design_structure]`
- **C: Subpopulations** -> `[task_subpopulations]`
- **D: Collection events** -> `[task_collection_events]`
- **E: Population** -> `[task_population]`
- **F: Contributors** -> `[task_contributors]`
- **G: Access conditions** -> `[task_access_conditions]`
- **H: Information** -> `[task_information]`
- **X1: Datasets** -> `[task_datasets]`
- **X2: Samplesets** -> `[task_samplesets]`
- **X3: Areas of information** -> `[task_areas_of_information]`
- **Y: Linkage** -> `[task_linkage]`

Each pass has:

- `template_json` (strict JSON schema)
- `instructions` (exact extraction rules)

If you want to change what is extracted or how strict the rules are, **edit only this file**.

---

## 2) `src/main_final.py` (the extractor)

This script:

1. Loads `config.final.toml` (or `PDF_EXTRACT_CONFIG`).
2. Runs the selected passes against the PDF text.
3. Writes a multi-sheet Excel file with consistent columns.

### Key behavior

- Uses `extract_pipeline.extract_fields()` for each pass.
- Outputs **one Excel file** with sheets:
  - `resources`
  - `subpopulations`
  - `collection_events`
  - `datasets`
  - `samplesets`
  - `organisations`
  - `people`
  - `publications`
  - `documentation`

### Run locally

```
python3 src/main_final.py -p all -o final_result.xlsx
```

### Run specific passes

```
python3 src/main_final.py -p A B E -o out.xlsx
```

### Run multiple PDFs

```
python3 src/main_final.py -p all \
  --pdfs data/a.pdf data/b.pdf \
  --paper-names paper_a paper_b \
  -o out.xlsx
```

---

## 3) `src/run_cluster_final.sh` (cluster workflow)

This script is the **production cluster runner**. It:

1. Starts two `llama-server` instances (GPU 0 and GPU 1).
2. Waits for `/health` to be OK.
3. Runs a TCP load balancer on port 18000.
4. Creates `config.runtime.toml` with `base_url` pointing to the load balancer.
5. Runs `src/main_final.py` with your args.
6. Copies the Excel output to your local destination (rsync).

### Default usage

```
bash src/run_cluster_final.sh
```

### Custom usage (passes, output, PDFs)

```
bash src/run_cluster_final.sh -p A -o out.xlsx --pdfs data/concrete.pdf data/oncolifes.pdf
```

### Important settings inside the script

- `LLAMA_BIN` -> path to `llama-server`
- `MODEL_PATH` -> GGUF model
- `PORT_LB` / `PORT_GPU0` / `PORT_GPU1`
- `CTX`, `SLOTS`, `NGL`
- `LOCAL_RSYNC_DEST` (where results are synced)

If you move models or change servers, update these variables at the top of the script.

---

## Typical workflow

1. Edit extraction rules in `config.final.toml`.
2. Run locally for quick tests: `python3 src/main_final.py -p A -o out.xlsx`.
3. Run on cluster for full output: `bash src/run_cluster_final.sh -p all -o final_result.xlsx`.

---

## Troubleshooting

- **No output / empty fields**: check `instructions` in `config.final.toml` for overly strict rules.
- **Wrong model / server**: update `[llm]` in `config.final.toml` or use `PDF_EXTRACT_CONFIG`.
- **Cluster failure**: check `logs/gpu0.log`, `logs/gpu1.log`, `logs/lb.log`.
- **PDF not found**: update `[pdf].path` or pass `--pdfs` to `main_final.py`.
