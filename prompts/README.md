# Prompts

The runtime now uses a single baseline prompt file:

- `prompts_cohort.toml`: baseline cohort prompt set

## Baseline and History

Prompt history is stored under `history/`:

```text
history/
  baseline/
    prompts_cohort.toml
  YYYY/
    YYYY-MM-DD/
      <timestamp>_run<run_id>.prompt_change.md
```

Meaning:

- `prompts_cohort.toml` is the editable baseline source-of-truth prompt file.
- `history/baseline/prompts_cohort.toml` is the stored baseline snapshot.
- `history/YYYY/YYYY-MM-DD/<timestamp>_run<run_id>.prompt_change.md` stores one compact prompt change artifact per run that actually changed the prompt.

If multiple prompt changes happen on the same day, they are stored in the same date directory.

## Which Prompt File Is Used

There are two important runtime cases.

### Default production route

- starts from `prompts_cohort.toml`
- runs prompt schema sync
- writes the effective prompt to `logs/runs/<run_id>/prompts.runtime.toml`

### Fully dynamic prompt route

- skips prompt schema sync
- generates task sections directly from the live EMX2 schema
- writes the generated prompt file to `<output>.dynamic_prompts.toml`

## Archive Directory

`archive/` contains older cohort prompt variants kept only for comparison or recovery. It is not a runtime entrypoint.
