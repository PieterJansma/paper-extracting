# Paper Extracting (Final Pipeline)

This repository extracts structured metadata from papers (PDFs) into a multi-sheet Excel file.
The final pipeline is built around:

- `config.final.toml` (prompts, schemas, extraction rules)
- `src/main_final.py` (multi-pass extraction + Excel writer)
- `src/run_cluster_final.sh` (cluster runtime with 2x llama-server + load balancer)

## What It Produces

One Excel workbook with these sheets:

- `resources`
- `subpopulations`
- `collection_events`
- `datasets`
- `samplesets`
- `organisations`
- `people`
- `publications`
- `documentation`

## Quick Start (Local)

1. Create and activate a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the project (recommended).

```bash
pip install -e . --no-build-isolation
```

3. Configure `config.final.toml`:
- `[llm]` -> your OpenAI-compatible endpoint (`base_url`, `model`, `api_key`)
- `[pdf].path` -> default PDF path
- `[pdf].max_pages = 0` -> read full paper (recommended for completeness)

4. Run all passes on one PDF.

```bash
pdf-extract -p all -o final_result.xlsx
```

## Common CLI Usage

Run selected passes:

```bash
pdf-extract -p A B E -o out.xlsx
```

Run multiple PDFs:

```bash
pdf-extract -p all \
  --pdfs data/a.pdf data/b.pdf \
  --paper-names paper_a paper_b \
  -o out.xlsx
```

Show CLI help:

```bash
pdf-extract --help
```

## Cluster Usage

`src/run_cluster_final.sh` does the full runtime setup:

- starts 2 llama-server instances (GPU0/GPU1)
- waits for health checks
- starts a local TCP load balancer
- writes `config.runtime.toml` with LB URL
- runs `src/main_final.py`
- syncs output if `LOCAL_RSYNC_DEST` is usable

Default run:

```bash
bash src/run_cluster_final.sh
```

All PDFs in `data/`:

```bash
bash src/run_cluster_final.sh -p all --pdfs data/*.pdf -o final_all.xlsx
```

## Key Behavior and Defaults

- `config.final.toml` is the source of truth for extraction logic.
- Long papers are handled with automatic chunked extraction fallback.
- Prompt cache is automatically disabled for very long papers when chunking is active.
- In health context, post-processing keeps HRI defaults consistent:
  - `theme` includes `Health`
  - `applicable_legislation` includes `Data Governance Act`
- Output is deterministic in column order (template-driven).

Long-paper tuning (optional, in `[llm]`):
- `chunking_enabled`
- `long_text_threshold_chars`
- `chunk_size_chars`
- `chunk_overlap_chars`
- `max_chunks` (optional cap)

## Files You Will Most Often Edit

- `config.final.toml`
  - change extraction instructions and templates
- `src/main_final.py`
  - change post-processing, fallbacks, and output logic
- `src/run_cluster_final.sh`
  - change model path, llama binary path, ports, rsync destination

## Troubleshooting

- No output or very sparse output:
  - prompts may be too strict; relax rules in `config.final.toml`
- Cluster model startup fails:
  - check `logs/gpu0.log`, `logs/gpu1.log`, `logs/lb.log`
- Wrong endpoint/model at runtime:
  - verify `[llm]` in `config.final.toml` or `PDF_EXTRACT_CONFIG`
- PDF not found:
  - fix `[pdf].path` or use `--pdfs`
- Scanned/image PDF gives almost no text:
  - OCR fallback runs only when extracted text is under 3000 chars
  - install `ocrmypdf` on cluster/node to enable OCR fallback
- Excel writer error:
  - install `openpyxl` or `xlsxwriter`

## Typical Workflow

1. Update extraction logic in `config.final.toml`.
2. Test quickly on 1 PDF locally.
3. Run full batch on cluster.
4. Review output workbook and audit issues.
