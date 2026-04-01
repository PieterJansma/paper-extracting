# Schemas

- `molgenis_UMCGCohortsStaging.csv`: profielgedreven schema-export voor de dynamische cohort promptflow.
- `molgenis_UMCGCohortsStaging.xlsx`: workbook voor post-write datatype-normalisatie.
- `molgenis_DataCatalogueFlat.csv`: extra catalogusreferentie die in de repo bewaard blijft, maar niet de standaard cohort prompt-sync aanstuurt.

De dynamische cohort-route start vanuit `schemas/molgenis_UMCGCohortsStaging.csv` en haalt daarna alleen de benodigde live EMX2 ontology/ref-bestanden op.
