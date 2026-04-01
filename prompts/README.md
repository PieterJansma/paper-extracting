# Prompts

- `prompts.toml`: baseline promptset voor de algemene/final pipeline.
- `prompts_cohort.toml`: cohort-specifieke baseline promptset.
- `archive/`: oudere of experimentele promptvarianten die niet de standaard runtime vormen.

De dynamische cohort-route gebruikt `prompts/prompts_cohort.toml` als startpunt en past alleen de tasks aan die geraakt worden door schema- of ontology-wijzigingen.
