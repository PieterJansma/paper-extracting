# paper-extracting

`paper-extracting` is the active cohort extraction pipeline in this repository. It extracts structured cohort metadata from one or more PDF papers, runs a local `llama.cpp`-based inference setup on a Slurm GPU node, fetches live EMX2 schema and ontology inputs, optionally uses a shared OCR model for weak PDFs, and writes a cohort workbook plus detailed run artifacts.

This README explains the complete workflow:

1. Sync the repository to the cluster.
2. Sync one or more PDF files to the cluster.
3. Build and download the cluster runtime with `setup_cluster_runtime.sbatch`.
4. Start a Slurm extraction job with `run_cluster_cohort.sbatch`.
5. Wait for the workbook and run artifacts.
6. Sync the workbook back to your local machine.


## Quick Start

If you already have cluster access and just want the shortest working route, use this sequence.

These examples assume:

- local repo path: `/Users/p.jansma/Documents/GitHub/paper-extracting/`
- cluster repo path: `/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/paper-extracting/`
- SSH host alias: `tunnel+nibbler`

Adjust those paths if your own setup differs.

### 1. Sync the repository to the cluster

```bash
rsync -avhP \
  /Users/p.jansma/Documents/GitHub/paper-extracting/ \
  tunnel+nibbler:/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/paper-extracting/
```

### 2. Sync one PDF to the cluster

```bash
rsync -avhP \
  /Users/p.jansma/Documents/GitHub/paper-extracting/data/oncolifes.pdf \
  tunnel+nibbler:/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/paper-extracting/data/
```

If your PDF lives somewhere else locally, sync that file instead.

### 3. Log in and move into the project directory

```bash
ssh tunnel+nibbler
cd /groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/paper-extracting
```

### 4. Preferred route: submit runtime setup as a Slurm batch job

```bash
sbatch --export=ALL,WORKDIR=/groups/umcg-gcc/tmp02/users/umcg-pjansma,BUILD_LLAMA=1 setup_cluster_runtime.sbatch
```

This is the standard and preferred setup route for this project. It runs `setup_cluster_runtime.sh` on a GPU node and prepares a git-ignored runtime inside this repo.

By default it creates:

- `./.runtime/llama.cpp`
- `./.runtime/GGUF/gemma-4-31B-it-Q4_K_M.gguf`
- `./.venv.cluster/`

### 5. Wait for setup to finish

```bash
watch -n 10 "squeue -u $USER"
```

You can inspect the setup job logs with:

```bash
ls -1 setup_cluster_runtime_*.out setup_cluster_runtime_*.err | tail
```

### 6. Start a small extraction test

```bash
sbatch --export=ALL run_cluster_cohort.sbatch -p all --pdfs data/oncolifes.pdf -o oncolifes_cohort.xlsx
```

This is the normal batch route for one PDF. The workbook will be written in the repo root on the cluster as:

```bash
oncolifes_cohort.xlsx
```

The detailed run artifacts are written under:

```bash
logs/runs/<run_id>/
```

### 7. Wait for the extraction job

```bash
watch -n 10 "squeue -u $USER"
```

The Slurm wrapper logs normally appear as:

```bash
cohort_extract-<jobid>.out
cohort_extract-<jobid>.err
```

### 8. Sync the workbook back to your local machine

```bash
rsync -avhP \
  tunnel+nibbler:/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/paper-extracting/oncolifes_cohort.xlsx \
  /Users/p.jansma/Documents/cluster_data/
```

### 9. Inspect the run artifacts if needed

```bash
ls -1dt /groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/paper-extracting/logs/runs/* | head
```

Inside the newest run directory, the first files to inspect are usually:

- `status.jsonl`
- `pipeline_issues.json`
- `config.runtime.toml`
- `prompts.runtime.toml`


## What This Project Does

`src/run_cluster_cohort.sh` is the supported runtime wrapper around the cohort extraction pipeline. In a normal production run it does the following:

1. Creates a run directory under `logs/runs/<run_id>/`.
2. Copies `config.cohort.toml` to a run-specific `config.runtime.toml`.
3. Builds the run-specific prompt file `prompts.runtime.toml`.
4. Fetches the latest EMX2 ontology and model CSV files from `molgenis/molgenis-emx2`.
5. Builds a live dynamic EMX2 runtime registry for the `UMCGCohortsStaging` profile.
6. Optionally schema-syncs the prompt file against the live EMX2 schema.
7. Starts two local `llama-server` instances and a local TCP load balancer.
8. Optionally starts the OCR `llama-server` and prefetches OCR text for weak PDFs.
9. Runs `src/main_cohort.py` to extract the selected PDF files into one cohort workbook.
10. Writes the workbook, run logs, prompt artifacts, OCR artifacts, and issue files.
11. Optionally syncs the workbook elsewhere with `rsync`.


## How To Start This Project On The Cluster

The recommended way to start this project is:

1. Sync the repository to the cluster.
2. Sync your PDF file or files to the cluster.
3. Log in to the cluster.
4. Move into the repository directory.
5. Submit `setup_cluster_runtime.sbatch`.
6. Wait until setup finishes.
7. Submit `run_cluster_cohort.sbatch`.

### Log in and enter the project directory

```bash
ssh tunnel+nibbler
cd /groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/paper-extracting
```

### Preferred setup route: `setup_cluster_runtime.sbatch`

This is the normal route for this project. Run:

```bash
sbatch --export=ALL,WORKDIR=/groups/umcg-gcc/tmp02/users/umcg-pjansma,BUILD_LLAMA=1 setup_cluster_runtime.sbatch
```

This wrapper job:

- allocates a GPU node through Slurm
- loads the required modules
- runs `setup_cluster_runtime.sh`
- clones `ggml-org/llama.cpp`
- checks out the pinned working commit used by this project
- optionally builds `llama.cpp`
- downloads and verifies the default Gemma GGUF used by this repo
- creates or updates `./.venv.cluster/`
- installs the project plus `pypdfium2`, `pillow`, and `xlsxwriter`

For most users, this is the correct way to do setup.

### What `setup_cluster_runtime.sh` does not do

It does not download the shared OCR model used by default in `src/run_cluster_cohort.sh`. The OCR defaults still point to the existing shared cluster paths:

- `/groups/umcg-gcc/tmp02/users/umcg-pjansma/Models/GGUF/GLM-OCR/GLM-OCR-Q8_0.gguf`
- `/groups/umcg-gcc/tmp02/users/umcg-pjansma/Models/GGUF/GLM-OCR/mmproj-GLM-OCR-Q8_0.gguf`

If those files are unavailable on your cluster, either:

- set `OCR_VLM_ENABLE=0`, or
- override `OCR_VLM_MODEL_PATH`, `OCR_VLM_MMPROJ_PATH`, and `OCR_VLM_LLAMA_BIN`

### Alternative manual route: interactive `srun`

If you want to build `llama.cpp` manually, do not do that on the login node. Start an interactive compute session first:

```bash
srun --cpus-per-task=4 --mem=32G --nodes=1 --gres=gpu:a40:2 --time=04:00:00 --pty bash -i
```

### Run setup inside that interactive session

Once you are inside the interactive compute shell, load the modules and run setup:

```bash
module purge
module load GCCcore/11.3.0
module load CMake/3.23.1-GCCcore-11.3.0
module load CUDA/12.2.0
module load Python/3.10.4-GCCcore-11.3.0

WORKDIR=/groups/umcg-gcc/tmp02/users/umcg-pjansma \
BUILD_LLAMA=1 \
bash setup_cluster_runtime.sh
```

This manual route does the same core setup work, but `setup_cluster_runtime.sbatch` remains the preferred route.

If `llama.cpp` is already built and you only want to refresh the venv and model file, you can run:

```bash
WORKDIR=/groups/umcg-gcc/tmp02/users/umcg-pjansma \
BUILD_LLAMA=0 \
bash setup_cluster_runtime.sh
```

### Separate isolated test runtime

If you want a separate test runtime instead of `./.runtime/` and `./.venv.cluster/`, use:

```bash
SETUP_SUFFIX=_test
```

That creates:

- `./.runtime_test/`
- `./.venv.cluster_test/`

Example:

```bash
SETUP_SUFFIX=_test BUILD_LLAMA=1 bash setup_cluster_runtime.sh
```


## Repository Layout

The active cohort application now lives directly in the repository root and `src/`.

Important files:

- `setup_cluster_runtime.sh`: repo-local runtime bootstrap for `llama.cpp`, Gemma, and the cluster venv
- `setup_cluster_runtime.sbatch`: preferred Slurm wrapper for runtime bootstrap
- `run_cluster_cohort.sbatch`: preferred Slurm extraction entrypoint
- `src/run_cluster_cohort.sh`: cluster runtime wrapper that starts the servers and runs the extraction
- `src/main_cohort.py`: main PDF-to-workbook extraction program
- `config.cohort.toml`: base runtime config
- `prompts/prompts_cohort.toml`: baseline prompt set
- `src/emx2_dynamic_runtime.py`: live EMX2 schema and ontology runtime builder
- `src/cohort_prompt_schema_updater.py`: prompt schema sync logic


## What You Need

Before you start, make sure you have all of the following.

### 1. A Slurm cluster with NVIDIA GPUs

The current scripts assume a GPU cluster and a Slurm scheduler.

The setup batch script currently requests:

- `1` node
- `2` A40 GPUs
- `4` CPUs
- `32G` memory
- `04:00:00` walltime

The extraction batch script currently requests:

- `1` node
- `1` task
- `2` A40 GPUs
- `4` CPUs
- `32G` memory
- `01:00:00` walltime

If your cluster uses different GPU names, partitions, or time limits, edit:

- `setup_cluster_runtime.sbatch`
- `run_cluster_cohort.sbatch`

### 2. A local machine with SSH and `rsync`

You need local shell access to:

- sync this repo to the cluster
- sync your PDF files to the cluster
- sync the resulting workbook back to your machine

### 3. A working OCR strategy

By default the runner expects the shared GLM-OCR paths shown above. If those are missing, either disable OCR or point to your own OCR model files.

### 4. One or more PDF files

The pipeline extracts structured cohort information from PDF papers. You must place the selected PDFs on the cluster before starting a run.


## Current Runtime Defaults

The current defaults are defined in `setup_cluster_runtime.sh` and `src/run_cluster_cohort.sh`.

### `llama.cpp`

Default binary:

```bash
./.runtime/llama.cpp/build/bin/llama-server
```

### Main extraction model

Default path:

```bash
./.runtime/GGUF/gemma-4-31B-it-Q4_K_M.gguf
```

The same model is started on both GPUs by default.

### Virtual environment

Preferred default:

```bash
./.venv.cluster/
```

Fallback if present:

```bash
./.venv/
```

### OCR defaults

Default OCR model:

```bash
/groups/umcg-gcc/tmp02/users/umcg-pjansma/Models/GGUF/GLM-OCR/GLM-OCR-Q8_0.gguf
```

Default OCR `mmproj`:

```bash
/groups/umcg-gcc/tmp02/users/umcg-pjansma/Models/GGUF/GLM-OCR/mmproj-GLM-OCR-Q8_0.gguf
```

Default OCR `llama-server`:

```bash
/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/llama.cpp-glmtest/build/bin/llama-server
```

### Default ports

- load balancer: `18000`
- GPU server 0: `18080`
- GPU server 1: `18081`
- OCR server: `18090`

### Default EMX2 source

- repo: `molgenis/molgenis-emx2`
- ref: `main`
- profile: `UMCGCohortsStaging`

### Important note about legacy log wording

Some runtime status messages still say `qwen_starting` and `qwen_ready`. That is only a leftover status label. The current default `MODEL_PATH` is Gemma, not Qwen.


## Step 1: Sync This Project To The Cluster

From your local machine, sync the repository to the cluster.

Example:

```bash
rsync -avhP \
  /Users/p.jansma/Documents/GitHub/paper-extracting/ \
  tunnel+nibbler:/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/paper-extracting/
```

Then log in and move into the project directory:

```bash
ssh tunnel+nibbler
cd /groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/paper-extracting
```


## Step 2: Sync Your PDF Files To The Cluster

You must also place the PDF files on the cluster before running.

Example for one file:

```bash
rsync -avhP \
  /Users/p.jansma/Documents/GitHub/paper-extracting/data/oncolifes.pdf \
  tunnel+nibbler:/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/paper-extracting/data/
```

Example for multiple files:

```bash
rsync -avhP \
  /Users/p.jansma/Documents/GitHub/paper-extracting/data/ \
  tunnel+nibbler:/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/paper-extracting/data/
```


## Step 3: Prepare The Runtime

The preferred route is:

```bash
sbatch --export=ALL,WORKDIR=/groups/umcg-gcc/tmp02/users/umcg-pjansma,BUILD_LLAMA=1 setup_cluster_runtime.sbatch
```

Wait for the job:

```bash
watch -n 10 "squeue -u $USER"
```

Inspect setup logs:

```bash
ls -1 setup_cluster_runtime_*.out setup_cluster_runtime_*.err | tail
```

If setup completes successfully, you should now have:

- `./.runtime/llama.cpp/build/bin/llama-server`
- `./.runtime/GGUF/gemma-4-31B-it-Q4_K_M.gguf`
- `./.venv.cluster/bin/python`


## Step 4: Start An Extraction Job

The main production route is the Slurm wrapper:

```bash
sbatch --export=ALL run_cluster_cohort.sbatch -p all --pdfs data/oncolifes.pdf -o oncolifes_cohort.xlsx
```

If you call `run_cluster_cohort.sbatch` without extra arguments, it uses a safe default:

```bash
sbatch run_cluster_cohort.sbatch
```

That default run becomes roughly:

```bash
bash src/run_cluster_cohort.sh -p all --pdfs data/oncolifes.pdf -o cohort_<jobid>.xlsx
```

### Multiple PDFs

Example:

```bash
sbatch --export=ALL run_cluster_cohort.sbatch \
  -p all \
  --pdfs data/oncolifes.pdf data/concrete.pdf \
  --paper-names oncolifes concrete \
  -o multi_cohort.xlsx
```

### Specific passes only

Example:

```bash
sbatch --export=ALL run_cluster_cohort.sbatch \
  -p A C E \
  --pdfs data/oncolifes.pdf \
  -o oncolifes_partial.xlsx
```

### Force OCR prefetch

Example:

```bash
sbatch --export=ALL run_cluster_cohort.sbatch \
  --ocr \
  -p all \
  --pdfs data/oncolifes.pdf \
  -o oncolifes_ocr.xlsx
```

### Force OCR text to be used

Example:

```bash
sbatch --export=ALL run_cluster_cohort.sbatch \
  --ocr-force-use \
  -p all \
  --pdfs data/oncolifes.pdf \
  -o oncolifes_ocr_forced.xlsx
```

### Write OCR compare files

Example:

```bash
sbatch --export=ALL run_cluster_cohort.sbatch \
  --ocr-dump \
  -p all \
  --pdfs data/oncolifes.pdf \
  -o oncolifes_ocr_dump.xlsx
```

### Disable OCR entirely

If the shared OCR model paths are unavailable or you do not want OCR:

```bash
sbatch --export=ALL,OCR_VLM_ENABLE=0 run_cluster_cohort.sbatch \
  -p all \
  --pdfs data/oncolifes.pdf \
  -o oncolifes_no_ocr.xlsx
```


## Step 5: Monitor The Job

### See all of your jobs

```bash
squeue -u $USER
```

### Refresh every 10 seconds

```bash
watch -n 10 "squeue -u $USER"
```

### See elapsed time and time left

```bash
watch -n 10 'squeue -u $USER -o "%.18i %.20j %.8T %.10M %.10L %.6D %R"'
```

### Inspect finished jobs

```bash
sacct -j <jobid> --format=JobID,JobName%30,State,ExitCode,Elapsed
```

### Inspect Slurm wrapper logs

```bash
tail -f cohort_extract-<jobid>.out cohort_extract-<jobid>.err
```

### Inspect run artifacts

The extraction wrapper creates a run directory under:

```bash
logs/runs/<run_id>/
```

The first files to inspect there are usually:

- `status.jsonl`
- `pipeline_issues.json`
- `config.runtime.toml`
- `prompts.runtime.toml`

If prompt schema sync ran, you may also see:

- `prompt_schema_sync.report.json`
- `prompt_schema_sync.compare.md`
- `prompt_schema_sync.prompt.diff`
- `prompt_schema_sync.before_after.md`

If OCR prefetch or OCR dumps ran, you may also see:

- `ocr_prefetch/`
- `text_compare/`
- `logs/ocr_prefetch_failed.txt`


## Step 6: Sync The Workbook Back To Your Local Machine

The safest default is to pull the workbook manually from your local machine after the job finishes.

Example:

```bash
rsync -avhP \
  tunnel+nibbler:/groups/umcg-gcc/tmp02/users/umcg-pjansma/Repositories/paper-extracting/oncolifes_cohort.xlsx \
  /Users/p.jansma/Documents/cluster_data/
```

If you wrote to a different output filename, pull that file instead.

### Optional automatic `rsync` from inside the cluster job

The runner supports automatic sync through:

- `LOCAL_RSYNC_DEST`
- `LOCAL_RSYNC_HOST`
- `SYNC_OUTPUT_ENABLE`
- `SYNC_REQUIRED`

Use that only if your cluster node can actually reach the target receiver. If that connectivity is uncertain, the manual pull route above is more reliable.


## Direct Interactive Run

If you are already inside an interactive compute session and want to run without the Slurm wrapper script, use:

```bash
bash src/run_cluster_cohort.sh -p all --pdfs data/oncolifes.pdf -o oncolifes_cohort.xlsx
```

That is useful for debugging, but `run_cluster_cohort.sbatch` is the normal production route.


## Output Files

The main output is the Excel workbook you pass with `-o`.

Example:

```bash
oncolifes_cohort.xlsx
```

The run directory always contains:

- `logs/runs/<run_id>/config.runtime.toml`
- `logs/runs/<run_id>/status.jsonl`
- `logs/runs/<run_id>/pipeline_issues.json`

The run directory usually also contains:

- `logs/runs/<run_id>/prompts.runtime.toml`
- `logs/runs/<run_id>/logs/`

Additional output files may be written next to the workbook:

- `<output>.dynamic_prompts.toml`
- `<output>.dynamic_emx2_registry.json`
- `<output>.dynamic_prompt_constraints.json`
- `<output>.issues.json`


## Common Overrides

### Override the runtime root

```bash
RUNTIME_ROOT=/path/to/custom_runtime bash setup_cluster_runtime.sh
```

### Override the venv path

```bash
VENV_DIR=/path/to/custom_venv bash setup_cluster_runtime.sh
```

### Override the `llama.cpp` binary for a run

```bash
LLAMA_BIN=/path/to/llama-server \
bash src/run_cluster_cohort.sh -p all --pdfs data/oncolifes.pdf -o out.xlsx
```

### Override the extraction model for a run

```bash
MODEL_PATH=/path/to/model.gguf \
bash src/run_cluster_cohort.sh -p all --pdfs data/oncolifes.pdf -o out.xlsx
```

### Use a different EMX2 fork or ref

```bash
MOLGENIS_EMX2_REPO="PieterJansma/molgenis-emx2" \
MOLGENIS_EMX2_REF="main" \
sbatch --export=ALL run_cluster_cohort.sbatch -p all --pdfs data/oncolifes.pdf -o fork_test.xlsx
```

### Use fully dynamic prompts for testing

```bash
COHORT_PROMPT_SCHEMA_SYNC=0 \
COHORT_DYNAMIC_PROMPTS=1 \
sbatch --export=ALL run_cluster_cohort.sbatch -p all --pdfs data/oncolifes.pdf -o dynamic_only.xlsx
```


## Practical Advice

- Use `setup_cluster_runtime.sbatch` as the normal setup route.
- Do not build `llama.cpp` on the login node.
- After code-only changes, syncing the repo is usually enough because the venv is installed in editable mode.
- After dependency changes in `pyproject.toml`, rerun `setup_cluster_runtime.sh` or `setup_cluster_runtime.sbatch`.
- If `llama-server` is missing, rerun setup with `BUILD_LLAMA=1`.
- If the Gemma file is missing, rerun setup and inspect the setup logs.
- If OCR fails because the shared GLM-OCR files are unavailable, set `OCR_VLM_ENABLE=0` or point to your own OCR model files.
- If the workbook sync fails, the workbook still remains on the cluster in the project directory unless you wrote it somewhere else explicitly.


## References

- `llama.cpp repository`: `https://github.com/ggml-org/llama.cpp`
- `Gemma 4 31B IT GGUF`: `https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF`
- `molgenis-emx2`: `https://github.com/molgenis/molgenis-emx2`
