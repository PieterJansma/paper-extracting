# Prompt Schema Demo

Deze fixture laat de schema-diff promptflow zien met een kleine nep-EMX2 bron die dezelfde mapstructuur gebruikt als de live fetch:

- `base_repo/`: basisversie
- `variant_repo/`: schema- en ontology-variant

De demo is expres klein en focust op vier cohort-taken:

- `task_access_conditions`: gewijzigd
  - veld toegevoegd: `data access url`
  - veld verwijderd: `release description`
  - ontology-keuzes gewijzigd: `Access rights`, `Release types`
- `task_collection_events`: gewijzigd
  - ontology-keuzes gewijzigd: `Data categories`
- `task_overview`: ongewijzigd
- `task_contributors_people`: ongewijzigd

Run de demo met:

```bash
bash src/run_prompt_schema_demo.sh
```

Belangrijkste outputs:

- `tmp/prompt_schema_demo/base_dynamic.toml`
- `tmp/prompt_schema_demo/variant_dynamic.toml`
- `tmp/prompt_schema_demo/updated_from_existing.toml`
- `tmp/prompt_schema_demo/prompt_update.compare.md`

De runner gebruikt bewust de bestaande `prompts_cohort.toml` als basisprompt en vervangt alleen de tasks die echt geraakt zijn door de variant.
