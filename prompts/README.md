# Prompts

De runtime gebruikt nog maar één baseline promptset:

- `prompts_cohort.toml`: cohort baseline promptset

Promptwijzigingen worden blijvend opgeslagen onder `history/`:

```text
history/
  baseline/
    prompts_cohort.toml
  YYYY/
    YYYY-MM-DD/
      <timestamp>_run<run_id>.prompt_change.md
```

`archive/` bevat oudere cohortvarianten voor vergelijking of herstel, maar is geen runtime-entrypoint.
